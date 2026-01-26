"""
Training entry point.

- Loads Hydra configuration
- Sets device, seed, and logging
- Initializes dataloaders
- Builds model from scratch or resumes from checkpoint
- Runs train / validation loops with AMP and grad accumulation
- Logs metrics and periodically saves checkpoint
"""
import torch
import logging
from contextlib import nullcontext
from pathlib import Path
import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from model import DETR, DETRConfig
from data import init_dataloaders, show_batch
from utils import (
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
    get_ist_time_now,
    timer,
    AverageMetrics,
    WandBLogger,
    DeTRMetrics
)
OmegaConf.register_new_resolver("now_ist", get_ist_time_now)

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg):
    logger = logging.getLogger("detr")
    device = torch_get_device(cfg.device_type)
    logger.info(f"Using {device}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = Path(hydra_cfg.runtime.output_dir)
    run_name = hydra_cfg.job.name
    torch_autocast_dtype = {'f32': torch.float32, 'bf16': torch.bfloat16}[cfg.autocast_dtype]

    torch_set_seed(cfg.rng_seed)

    train_dataloader = init_dataloaders(cfg, split='train')
    if cfg.interactive:
        show_batch(train_dataloader, N=16)
    val_dataloader = init_dataloaders(cfg, split = 'val')

    start_epoch = 1
    wandb_id = None
    if cfg.init_from == 'scratch':
        model_cfg = DETRConfig(**cfg.model)
        model = DETR(model_cfg)
        model.to(device)
    else:
        ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
        ckpt_cfg = ckpt['config']
        model_cfg = DETRConfig(**ckpt_cfg.model)
        model = DETR(model_cfg)
        model.to(device)
        model.load_state_dict(torch_compile_ckpt_fix(ckpt['model']))
        logger.info(f"Loaded checkpoint from {cfg.init_from}")
        start_epoch = ckpt['epoch'] + 1
        wandb_id = ckpt.get('wandb_id')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_cfg.name} | Params: {total_params / 1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")
    if cfg.torch_compile:
        model = torch.compile(model, dynamic=True)

    # optimizer
    optimizer = model.configure_optimizer(cfg.optimizer, device=device)
    lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[cfg.n_epochs - 100], gamma=0.1)
    if cfg.init_from != "scratch":
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_schdlr_state = ckpt.get('lr_scheduler', None)
        if lr_schdlr_state and lr_scheduler:
            lr_scheduler.load_state_dict(lr_schdlr_state)

    if cfg.enable_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch_autocast_dtype)
        if device.type == "cuda" and torch_autocast_dtype == torch.bfloat16
        else nullcontext()
    )

    wb_logger = WandBLogger(
        project=cfg.logging.wandb.project,
        run=run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        tags=("train", "val"),
        metrics=("loss", "loss_cls", "loss_l1", "loss_giou", "time", "cls_acc_matched", "mean_iou", "bg_ratio"),
        run_id=wandb_id,
        enable=cfg.logging.wandb.enable
    )

    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = cfg.grad_accum_steps

    @timer
    def train_epoch():
        model.train()
        loss_metrics = AverageMetrics()
        detr_metrics = DeTRMetrics(model_cfg.background_cls_idx)
        progress_bar = tqdm(train_dataloader, dynamic_ncols=True, desc="Train", leave=False, disable=(not cfg.interactive))
        for step, batch in enumerate(progress_bar):
            imgs, tgts = batch[0], batch[1]
            imgs = imgs.to(device)
            for tgt in tgts:
                tgt['labels'] = tgt['labels'].to(device)
                tgt['bboxes'] = tgt['bboxes'].to(device)

            with autocast_ctx:
                ret = model(imgs, tgts)
            pred_cls_probs = ret['class_probs']
            pred_bboxes = ret['bboxes']
            match_indices = ret['matched_indices']
            loss_i = ret['loss']['loss']

            (loss_i / grad_accum_steps).backward()
            if (step+1) % grad_accum_steps == 0:
                if cfg.clip_grad_norm_1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = imgs.size(0)
            loss_metrics.update({ k:v.item() for k, v in ret['loss'].items() }, batch_size)
            detr_metrics.update(pred_cls_probs, pred_bboxes, tgts, match_indices)
            progress_bar.set_postfix({ k:f"{v.item():.4f}" for k, v in ret['loss'].items() })

        if (step+1) % grad_accum_steps != 0:
            if cfg.clip_grad_norm_1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        progress_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return loss_metrics.compute() | detr_metrics.compute()

    @timer
    @torch.no_grad()
    def val_epoch():
        model.eval()
        loss_metrics = AverageMetrics()
        detr_metrics = DeTRMetrics(model_cfg.background_cls_idx)
        progress_bar = tqdm(val_dataloader, dynamic_ncols=True, desc="Val", leave=False, disable=(not cfg.interactive))
        for step, batch in enumerate(progress_bar):
            imgs, tgts = batch[0], batch[1]
            imgs = imgs.to(device)
            for tgt in tgts:
                tgt['labels'] = tgt['labels'].to(device)
                tgt['bboxes'] = tgt['bboxes'].to(device)

            with autocast_ctx:
                ret = model(imgs, tgts)
            pred_cls_probs = ret['class_probs']
            pred_bboxes = ret['bboxes']
            match_indices = ret['matched_indices']

            batch_size = imgs.size(0)
            loss_metrics.update({ k:v.item() for k, v in ret['loss'].items() }, batch_size)
            detr_metrics.update(pred_cls_probs, pred_bboxes, tgts, match_indices)
            progress_bar.set_postfix({ k:f"{v.item():.4f}" for k, v in ret['loss'].items() })

        progress_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return loss_metrics.compute() | detr_metrics.compute()

    t, stats = val_epoch()
    logger.info(f"Initial Val " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()) + f" Time={t:.2f}s")
    wb_logger.log("val", {'epoch': start_epoch-1, **stats, 'time':t})
    for epoch in range(start_epoch, cfg.n_epochs + 1):
        last_epoch = epoch == cfg.n_epochs
        logger.info(f"Epoch: {epoch}/{cfg.n_epochs}")

        t, stats = train_epoch()
        logger.info(f"{'Train':<5} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()) + f" Time={t:.2f}s")
        wb_logger.log("train", {'epoch': epoch, **stats, 'time':t})

        if lr_scheduler is not None:
            lr_scheduler.step()

        if last_epoch or epoch % cfg.val_every_epoch == 0:
            t, stats = val_epoch()
            logger.info(f"{'Val':<5} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()) + f" Time={t:.2f}s")
            wb_logger.log("val", {'epoch': epoch, **stats, 'time':t})

        if last_epoch or epoch % cfg.save_every_epoch == 0:
            ckpt_path = log_dir / f"{cfg.model.name}.pt"
            torch.save({
                'model': model.state_dict(),
                'config': cfg,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'wandb_id': wb_logger.run_id,
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {str(ckpt_path)}")


if __name__ == "__main__":
    main()