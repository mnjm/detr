"""
Utility functions
"""

import os
from collections import defaultdict
from datetime import datetime
from functools import wraps
from time import time
import pytz
import torch
from torchvision.ops import box_iou
import wandb
from dotenv import load_dotenv

def to_2tuple(x):
    return x if isinstance(x, (list, tuple)) else (x, x)

def torch_get_device(device_type):
    if device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available :(, `python train.py +device=auto`"
        device = torch.device("cuda")
    elif device_type == "auto":
        assert not torch.cuda.is_available(), "CUDA is available :), switch to cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore
                device = xm.xla_device()
            except ImportError:
                device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device

def torch_set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def torch_compile_ckpt_fix(state_dict):
    # when torch.compiled a model, state_dict is updated with a prefix '_orig_mod.', renaming this
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def get_ist_time_now(fmt="%d-%m-%Y-%H%M%S"):
    tz = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(tz)
    return now_ist.strftime(fmt)

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time()
        ret = func(*args, **kwargs)
        t = time() - t0
        return t, ret
    return wrapper

class AverageMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    def update(self, values, n=1):
        if n <= 0: return
        for k, v in values.items():
            self.total[k] += v * n
            self.count[k] += n

    def compute(self):
        return {k: self.total[k] / self.count[k] for k in self.count}

class DeTRMetrics:
    """
    Simple DETR metrics
    """

    def __init__(self, background_idx: int):
        self.bg_idx = background_idx
        self.reset()

    def reset(self):
        self.cls_correct = 0
        self.cls_total = 0
        self.iou_sum = 0.0
        self.iou_count = 0
        self.bg_preds = 0
        self.total_preds = 0

    @torch.no_grad()
    def update(self, cls_probs, bboxes, targets, match_indices):
        """
        Args:
            cls_probs: (B, Q, C) - class probabilities (post-softmax)
            bboxes: (B, Q, 4)
            targets: list of dicts
            match_indices: Hungarian output
        """
        B, Q, _ = cls_probs.shape

        preds_cls = cls_probs.argmax(dim=-1)

        self.bg_preds += (preds_cls == self.bg_idx).sum().item()
        self.total_preds += B * Q

        for i, (pred_idx, tgt_idx) in enumerate(match_indices):
            if tgt_idx.numel() == 0:
                continue

            pred_labels = preds_cls[i, pred_idx]
            tgt_labels = targets[i]["labels"][tgt_idx]

            self.cls_correct += (pred_labels == tgt_labels).sum().item()
            self.cls_total += tgt_labels.numel()

            pred_boxes = bboxes[i, pred_idx]
            tgt_boxes = targets[i]["bboxes"][tgt_idx]
            ious = box_iou(pred_boxes, tgt_boxes).diag()

            self.iou_sum += ious.sum().item()
            self.iou_count += ious.numel()

    def compute(self):
        return {
            "cls_acc_matched": (self.cls_correct / self.cls_total if self.cls_total > 0 else 0.0),
            "mean_iou": (self.iou_sum / self.iou_count if self.iou_count > 0 else 0.0),
            "bg_ratio": (self.bg_preds / self.total_preds if self.total_preds > 0 else 0.0),
        }

class WandBLogger:

    def __init__(self, project, run, config, tags, metrics, run_id=None, enable=False):
        self.enable = enable
        self.run_id = run_id
        if self.enable:
            load_dotenv()
            wnb_key = os.getenv('WANDB_API_KEY')
            assert wnb_key is not None, "WANDB_API_KEY not loaded in env"
            wandb.login(key=wnb_key)
            run = wandb.init(project=project, name=run, config=config, id=self.run_id, resume="allow")
            self.run_id = run.id
            self.tags = tags
            self.metrics = metrics
            wandb.define_metric("epoch")
            for tag in tags:
                for metric in metrics:
                    wandb.define_metric(f"{tag}/{metric}", step_metric="epoch")

    def log(self, tag, data):
        if self.enable:
            assert tag in self.tags, f"{tag=} not created"
            new_data = {'epoch': data['epoch']}
            for metric, val in data.items():
                if metric == "epoch":
                    continue
                assert metric in self.metrics, f"{metric=} not created"
                new_data[f"{tag}/{metric}"] = val
            wandb.log(new_data)