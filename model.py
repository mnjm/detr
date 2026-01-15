"""
Minimal Simplified version of DETR (DEtection TRansformer) Model Implementation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import FrozenBatchNorm2d, box_convert, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from einops import rearrange
from hydra.utils import instantiate
from dataclasses import dataclass

@dataclass
class DETRConfig:
    name: str = "detr-b"
    backbone: str = "resnet50"
    freeze_backbone: bool = False
    n_queries: int = 32
    background_cls_idx: int = 0
    hidden_dim: int = 256
    n_encoders: int = 6
    n_encoder_heads: int = 8
    n_decoders: int = 6
    n_decoder_heads: int = 8
    ffn_intermediate_dim: int = 2048
    n_class: int = 21
    attn_drop_rate: float = .0
    ffn_drop_rate: float = .0

    cls_matching_weight: float = 1.
    l1_matching_weight: float = 5.
    giou_matching_weight: float = 2.

    background_cls_weight: float = .1

def sinosoidal_2d_position_embeddings(H: int, W: int, dim: int, device: torch.device):
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        H: feature map height
        W: feature map width
        dim: Dimension of positional embedding (must be divisible by 4)
        device: Torch device for tensor creation

    Returns:
        Positional embeddings of shape (H*W, dim)
    """
    assert dim % 4 == 0, "dim must be divisible by 4"

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    y = y.flatten()
    x = x.flatten()

    dim_quarter = dim // 4
    omega = torch.arange(dim_quarter, device=device) / dim_quarter
    omega = 1.0 / (10000 ** omega)

    x = x[:, None] * omega[None, :]
    y = y[:, None] * omega[None, :]

    pos_x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    pos_y = torch.cat([torch.sin(y), torch.cos(y)], dim=1)

    pos = torch.cat([pos_x, pos_y], dim=1)
    return pos

class FFN(nn.Module):
    """FFN block."""

    def __init__(
            self, in_features: int, hidden_features: int, out_features: int,
            act_fn: type[nn.Module]=nn.ReLU, drop_rate: float=.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_fn()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class EncoderLayer(nn.Module):

    def __init__(
            self, dim: int, n_heads: int, ffn_intermediate_dim:int, attn_drop_rate: float=.0,
            ffn_drop_rate: float=.0
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=attn_drop_rate, batch_first=True
        )
        self.attn_dropout = nn.Dropout(attn_drop_rate)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FFN(
            in_features=dim, hidden_features=ffn_intermediate_dim, out_features=dim,
            drop_rate=ffn_drop_rate
        )

    def forward(self, x: torch.tensor, pos_emb: torch.tensor):
        nx = self.attn_norm(x)
        # In DETR Encoder, query and key is added with pos embedding
        attn_out, attn_weights = self.attn(
            query = nx + pos_emb,
            key = nx + pos_emb,
            value=nx
        )
        x = x + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class DecoderLayer(nn.Module):

    def __init__(
            self, dim: int, n_heads: int, ffn_intermediate_dim:int, attn_drop_rate: float=.0,
            ffn_drop_rate: float=.0
    ):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=attn_drop_rate, batch_first=True
        )
        self.attn_dropout = nn.Dropout(attn_drop_rate)
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=attn_drop_rate, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FFN(
            in_features=dim, hidden_features=ffn_intermediate_dim, out_features=dim,
            drop_rate=ffn_drop_rate
        )

    def forward(
            self, query_objs: torch.tensor, encoder_memory: torch.tensor, query_emb: torch.tensor,
            pos_emb: torch.tensor
        ):
        """
        Args:
            query_objs (tensor): (B, N, D)
            encoder_memory (tensor): (B, N, D)
            query_emb (tensor): (1, N, D)
            pos_emb (tensor): (1, N, D)
        """
        residual = query_objs
        # Self Attn
        x = self.self_attn_norm(query_objs)
        attn_out, _ = self.self_attn(
            query = x + query_emb,
            key = x + query_emb,
            value = x
        )
        residual = residual + self.attn_dropout(attn_out)
        # Cross Attn
        x = self.cross_attn_norm(residual)
        attn_out, _ = self.cross_attn(
            query = x + query_emb,
            key = encoder_memory + pos_emb,
            value = encoder_memory
        )
        residual = residual + self.attn_dropout(attn_out)
        # FFN
        x = self.ffn_norm(residual)
        x = self.ffn(x)
        return residual + x

class DETR(nn.Module):

    def __init__(self, cfg: DETRConfig):
        super().__init__()
        self.cfg = cfg

        backbone_map = {
            'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
            'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
            'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
            'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
        }
        assert cfg.backbone in backbone_map, f"Backbone should be one of {backbone_map.keys()}"
        backbone_fn, backbone_weights, backbone_chls = backbone_map[cfg.backbone]
        backbone = backbone_fn(weights=backbone_weights, norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        if cfg.freeze_backbone:
            self.backbone.requires_grad_(False)

        self.backbone_proj = nn.Conv2d(backbone_chls, cfg.hidden_dim, kernel_size=1)

        self.encoder = nn.ModuleList([
            EncoderLayer(
                dim=cfg.hidden_dim, n_heads=cfg.n_encoder_heads,
                ffn_intermediate_dim=cfg.ffn_intermediate_dim,
                attn_drop_rate=cfg.attn_drop_rate, ffn_drop_rate=cfg.ffn_drop_rate,
            ) for _ in range(cfg.n_encoders)
        ])

        self.query_embeddings = nn.Parameter(torch.randn(1, self.cfg.n_queries, cfg.hidden_dim))

        self.decoder = nn.ModuleList([
            DecoderLayer(
                dim=cfg.hidden_dim, n_heads=cfg.n_encoder_heads,
                ffn_intermediate_dim=cfg.ffn_intermediate_dim,
                attn_drop_rate=cfg.attn_drop_rate, ffn_drop_rate=cfg.ffn_drop_rate,
            ) for _ in range(cfg.n_decoders)
        ])

        self.class_ffn = nn.Linear(cfg.hidden_dim, cfg.n_class)
        self.bbox_ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 4)
        )

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if "backbone." in name:
                continue

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.tensor, targets: list[dict]=None):

        x = self.backbone(x)
        x = self.backbone_proj(x)

        B, D, H, W = x.shape
        x = rearrange(x, "b d h w -> b (h w) d")

        pos_emb = sinosoidal_2d_position_embeddings(H, W, D, x.device).unsqueeze(0) # 1, H*W, D
        for lyr in self.encoder:
            x = lyr(x, pos_emb)

        encoder_memory = x
        B, N, D = x.shape
        query_objs = torch.zeros((B, self.cfg.n_queries, D), device=x.device)

        decoder_outputs = []
        for lyr in self.decoder:
            query_objs = lyr(query_objs, encoder_memory, self.query_embeddings, pos_emb)
            if self.training:
                decoder_outputs.append(query_objs) # for aux losses

        if self.training:
            query_objs = torch.stack(decoder_outputs) # (n_layers, B, n_queries, D)
        else:
            query_objs = query_objs.unsqueeze(0) # (1, B, n_queries, D)

        cls_logits = self.class_ffn(query_objs) # ((n_layers or 1), B, n_queries, n_class)
        bboxs_output = self.bbox_ffn(query_objs).sigmoid() # ((n_layers or 1), B, n_queries, 4)
        bboxs_output = box_convert(boxes=bboxs_output, in_fmt='cxcywh', out_fmt='xyxy')

        ret = {
            'class_probs': cls_logits[-1].softmax(dim=-1),
            'bboxes': bboxs_output[-1]
        }

        if targets is not None:
            loss = self._loss_fn(cls_logits, bboxs_output, targets)
            ret['loss'] = loss

        return ret

    @torch._dynamo.disable()
    @torch.no_grad()
    def _hungarian_matching(self, cls_logits: torch.Tensor, bboxes: torch.Tensor, targets: list[dict]):
        """
        Perform Hungarian matching for entire batch at once.

        Args:
            cls_logits: (B, n_queries, n_class)
            bboxes: (B, n_queries, 4) in xyxy format
            targets: list of dicts with 'labels' and 'bboxes'

        Returns:
            list of tuples (pred_idx, target_idx) for each image in batch
        """
        B, n_queries, _ = cls_logits.shape
        cfg = self.cfg

        out_prob = cls_logits.softmax(dim=-1)  # (B*n_queries, n_cls)

        match_indices = []
        for i in range(B):
            # Slice predictions for this image
            out_prob_i = out_prob[i]
            out_bbox_i = bboxes[i]

            tgt_lbls_i = targets[i]["labels"]
            tgt_bbox_i = targets[i]["bboxes"]

            cost_class = -out_prob_i[:, tgt_lbls_i]
            cost_bbox = torch.cdist(out_bbox_i, tgt_bbox_i, p=1)
            cost_giou = -generalized_box_iou(out_bbox_i, tgt_bbox_i)

            C = (cfg.cls_matching_weight * cost_class +
                cfg.l1_matching_weight * cost_bbox +
                cfg.giou_matching_weight * cost_giou)

            C = C.cpu()
            pred_idx, target_idx = linear_sum_assignment(C.numpy())

            match_indices.append((
                torch.as_tensor(pred_idx, dtype=torch.int64, device=cls_logits.device),
                torch.as_tensor(target_idx, dtype=torch.int64, device=cls_logits.device)
            ))

        return match_indices

    def _loss_fn(self, cls_logits: torch.tensor, bboxes: torch.tensor, targets: list[dict]):
        """
        Calculates DETR loss with support for aux losses.

        Args:
            cls_logits: (n_layers, B, n_queries, n_class)
            bboxes: (n_layers, B, n_queries, 4)
            targets: list of dicts
        """
        n_layers, B, n_queries, n_class = cls_logits.shape
        cfg = self.cfg
        device = cls_logits.device

        loss_cls_total ,loss_l1_total ,loss_giou_total = .0, .0, .0

        for layer_idx in range(n_layers):
            logits_l = cls_logits[layer_idx]  # (B, n_queries, n_class)
            bboxes_l = bboxes[layer_idx]  # (B, n_queries, 4)
            match_indices = self._hungarian_matching(logits_l, bboxes_l, targets)

            pred_batch_idxs = torch.cat([
                torch.ones_like(pred_idx) * i
                for i, (pred_idx, _) in enumerate(match_indices)
            ])

            pred_query_idxs = torch.cat([pred_idx for pred_idx, _ in match_indices])

            valid_target_labels = torch.cat([
                targets[i]['labels'][target_idx]
                for i, (_, target_idx) in enumerate(match_indices)
            ])

            target_classes = torch.full(
                (B, n_queries),
                fill_value=cfg.background_cls_idx,
                dtype=torch.int64,
                device=device
            )

            target_classes[pred_batch_idxs, pred_query_idxs] = valid_target_labels

            cls_weights = torch.ones(n_class, device=device)
            cls_weights[cfg.background_cls_idx] = cfg.background_cls_weight

            loss_cls = F.cross_entropy(
                logits_l.reshape(-1, n_class),
                target_classes.reshape(-1),
                weight=cls_weights
            )

            matched_pred_boxes = bboxes_l[pred_batch_idxs, pred_query_idxs]

            matched_target_boxes = torch.cat([
                targets[i]['bboxes'][target_idx]
                for i, (_, target_idx) in enumerate(match_indices)
            ], dim=0)

            num_matched_boxes = matched_pred_boxes.shape[0]
            if num_matched_boxes > 0:
                loss_l1 = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='sum')
                loss_l1 = loss_l1 / num_matched_boxes

                # GIoU loss (averaged per matched box)
                loss_giou = (1.0 - generalized_box_iou(
                    matched_pred_boxes, matched_target_boxes
                ).diag()).sum()
                loss_giou = loss_giou / num_matched_boxes
            else:
                loss_l1 = torch.tensor(0.0, device=device)
                loss_giou = torch.tensor(0.0, device=device)

            loss_cls_total += loss_cls
            loss_l1_total += loss_l1
            loss_giou_total += loss_giou

        loss_cls_avg = loss_cls_total / n_layers
        loss_l1_avg = loss_l1_total / n_layers
        loss_giou_avg = loss_giou_total / n_layers

        loss = (cfg.cls_matching_weight * loss_cls_avg
                + cfg.l1_matching_weight * loss_l1_avg
                + cfg.giou_matching_weight * loss_giou_avg)

        return {
            'loss': loss,
            'loss_cls': loss_cls_avg,
            'loss_l1': loss_l1_avg,
            'loss_giou': loss_giou_avg
        }

    def configure_optimizer(self, optim_cfg: object, device: torch.device):
        supported_optimizers_map = { 'adamw': torch.optim.AdamW }
        assert optim_cfg.type in supported_optimizers_map, f"f{optim_cfg.name=} optimizer not supported"
        optim_init = supported_optimizers_map[optim_cfg.type]

        optim_cfg.fused = getattr(optim_cfg, 'fused', False) and device.type == "cuda"
        weight_decay = getattr(optim_cfg, 'weight_decay', 1e-2)
        lr = optim_cfg.lr
        optim_groups = []
        if not self.cfg.freeze_backbone:
            params_dict = { pn: p for pn, p in self.backbone.named_parameters() if p.requires_grad }
            # create optim groups of any params that is 2D or more. This group will be weight decayed ie weight tensors in Linear and embeddings
            decay_params = [ p for p in params_dict.values() if p.dim() >= 2 ]
            # create optim groups of any params that is 1D. All biases and layernorm params
            no_decay_params = [ p for p in params_dict.values() if p.dim() < 2 ]
            optim_groups.extend([
                { 'params': decay_params, 'weight_decay': weight_decay, 'lr': lr * .1 },
                { 'params': no_decay_params, 'weight_decay': .0, 'lr': lr * .1 }
            ])
        params_dict = { pn: p for pn, p in self.named_parameters() if 'backbone.' not in pn }
        params_dict = { pn:p for pn, p in params_dict.items() if p.requires_grad } # filter params that requires grad
        decay_params = [ p for p in params_dict.values() if p.dim() >= 2]
        no_decay_params = [ p for p in params_dict.values() if p.dim() < 2]
        weight_decay = getattr(optim_cfg, 'weight_decay', 0.0)
        optim_groups.extend([
            { 'params': decay_params, 'weight_decay': weight_decay, 'lr': lr },
            { 'params': no_decay_params, 'weight_decay': 0.0, 'lr': lr },
        ])
        kwargs = dict(optim_cfg)
        del kwargs['type']
        optimizer = optim_init(optim_groups, **kwargs)
        return optimizer