"""
model.py — Mask-Transformer 기반 3D Instance Segmentation (pure PyTorch)

Architecture
------------
Backbone : PointNet++ U-Net  (SA × 3  +  FP × 3)
Decoder  : Transformer decoder (self-attn + cross-attn to SA3 bottleneck)
Heads    : per-query binary mask  +  fg/bg classification

forward() returns
    'pred_logits' : [B, Q, 2]    — fg/bg score per query
    'pred_masks'  : [B, Q, N]    — per-point mask logit per query

run_inference() returns [B, N] long tensor  (0=bg, 1~100=instance)

Reference
---------
Mask3D: Schult et al., "Mask3D: Mask Transformer for 3D Instance Segmentation",
ICRA 2023.  https://arxiv.org/abs/2210.03105

Differences from the original Mask3D:
  - backbone: PointNet++ SA+FP instead of MinkowskiEngine sparse-conv Res16UNet
    (no custom CUDA required)
  - single cross-attention level (SA3 bottleneck, 512 pts) instead of
    multi-scale feature pyramid (simpler, fits within parameter budget)
  - single category (Nubzuki) → n_queries=20, num_classes=2
  - greedy NMS inference instead of learned mask scoring MLP
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


# ═══════════════════════════════════════════════════════════════
# Utility: CPU kNN (scipy) — shared by SA / FP
# ═══════════════════════════════════════════════════════════════

def _knn_indices(support: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(support))
    _, idx = cKDTree(support).query(query, k=k, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, np.newaxis]
    return idx.astype(np.int32)


# ═══════════════════════════════════════════════════════════════
# PointNet++ backbone building blocks
# ═══════════════════════════════════════════════════════════════

class SharedMLP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bn: bool = True, relu: bool = True):
        super().__init__()
        layers: list = [nn.Conv1d(in_ch, out_ch, 1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm1d(out_ch))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetAbstraction(nn.Module):
    """Random subsampling → kNN grouping → SharedMLP + maxpool."""

    def __init__(self, n_sample: int, k: int, in_ch: int, mlp_channels: list):
        super().__init__()
        self.n_sample = n_sample
        self.k = k
        layers, prev = [], in_ch + 3
        for ch in mlp_channels:
            layers.append(SharedMLP(prev, ch))
            prev = ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor):
        """xyz [B,3,N], feats [B,C,N]  →  new_xyz [B,3,S], new_feats [B,C_out,S]"""
        B, C, N = feats.shape
        S = min(self.n_sample, N)

        perm    = torch.randperm(N, device=xyz.device)[:S]
        new_xyz = xyz[:, :, perm]                                    # [B,3,S]

        idx = _knn_indices(
            xyz[0].T.detach().cpu().numpy(),
            new_xyz[0].T.detach().cpu().numpy(),
            self.k,
        )
        idx_t    = torch.from_numpy(idx.astype(np.int64)).to(xyz.device).reshape(-1)
        gathered = feats[:, :, idx_t].reshape(B, C, S, self.k)
        rel_xyz  = xyz[:, :, idx_t].reshape(B, 3, S, self.k) - new_xyz.unsqueeze(3)
        group    = torch.cat([gathered, rel_xyz], 1)                 # [B,C+3,S,k]

        group     = group.permute(0, 2, 1, 3).reshape(B * S, C + 3, self.k)
        new_feats = self.mlp(group).max(2)[0].reshape(B, S, -1).permute(0, 2, 1)
        return new_xyz, new_feats


class FeaturePropagation(nn.Module):
    """IDW interpolation coarse→fine, skip-concat, SharedMLP."""

    def __init__(self, in_ch: int, mlp_channels: list):
        super().__init__()
        layers, prev = [], in_ch
        for ch in mlp_channels:
            layers.append(SharedMLP(prev, ch))
            prev = ch
        self.mlp = nn.Sequential(*layers)

    def forward(self,
                xyz_fine:     torch.Tensor,
                xyz_coarse:   torch.Tensor,
                feats_fine:   torch.Tensor,
                feats_coarse: torch.Tensor) -> torch.Tensor:
        B, _, N = xyz_fine.shape
        M  = xyz_coarse.shape[2]
        Cc = feats_coarse.shape[1]

        k   = min(3, M)
        idx = _knn_indices(
            xyz_coarse[0].T.detach().cpu().numpy(),
            xyz_fine[0].T.detach().cpu().numpy(),
            k,
        )
        idx_t    = torch.from_numpy(idx.astype(np.int64)).to(xyz_fine.device).reshape(-1)
        nbr_pts  = xyz_coarse[0, :, idx_t].T.reshape(N, k, 3)
        dists    = torch.norm(xyz_fine[0].T.unsqueeze(1) - nbr_pts, dim=2).clamp(min=1e-8)
        w        = 1.0 / dists
        w        = w / w.sum(1, keepdim=True)

        nbr_f   = feats_coarse[0, :, idx_t].T.reshape(N, k, Cc)
        interp  = (w.unsqueeze(2) * nbr_f).sum(1).T.unsqueeze(0)   # [B,Cc,N]
        return self.mlp(torch.cat([interp, feats_fine], 1))


# ═══════════════════════════════════════════════════════════════
# Transformer decoder
# ═══════════════════════════════════════════════════════════════

class _DecoderLayer(nn.Module):
    """Self-attention → cross-attention → FFN  (Pre-LN style)."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # q   : [B, Q, d_model]
        # mem : [B, M, d_model]
        q = q + self.drop(self.self_attn(self.norm1(q), self.norm1(q), self.norm1(q),
                                         need_weights=False)[0])
        q = q + self.drop(self.cross_attn(self.norm2(q), mem, mem,
                                          need_weights=False)[0])
        q = q + self.drop(self.ffn(self.norm3(q)))
        return q


# ═══════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════

class MaskTransformerNet(nn.Module):
    """
    PointNet++ U-Net backbone + Transformer decoder + per-query mask prediction.

    Backbone (m=64):
      SA1: N  → 8192   in=9   mlp=[m, m, 2m]
      SA2: 8192→2048   in=2m  mlp=[2m, 2m, 4m]
      SA3: 2048→512    in=4m  mlp=[4m, 4m, 8m]   ← cross-attention keys/values
      FP3: 512 →2048   in=8m+4m  mlp=[4m, 4m]
      FP2: 2048→8192   in=4m+2m  mlp=[2m, 2m]
      FP1: 8192→N      in=2m+9   mlp=[2m, d_model]  ← mask features

    Decoder (d_model=128):
      query_embed  : n_queries × d_model  (learnable)
      sa3_proj     : 8m → d_model         (project bottleneck to d_model)
      3 × _DecoderLayer (self + cross + ffn)

    Heads:
      cls_head : d_model → 2   (fg/bg per query)
      mask     : queries [B,Q,d] · point_feats [B,d,N] = [B,Q,N]  (no extra params)
    """

    def __init__(self,
                 in_channels: int = 9,
                 num_classes: int = 2,
                 m: int = 64,
                 n1: int = 8192,
                 n2: int = 2048,
                 n3: int = 512,
                 k: int = 32,
                 n_queries: int = 20,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_decoder_layers: int = 3,
                 d_ffn: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self._d_model = d_model

        # ── Backbone ──
        self.sa1 = SetAbstraction(n1, k, in_channels,  [m,   m,   m * 2])
        self.sa2 = SetAbstraction(n2, k, m * 2,        [m*2, m*2, m * 4])
        self.sa3 = SetAbstraction(n3, k, m * 4,        [m*4, m*4, m * 8])

        self.fp3 = FeaturePropagation(m * 8 + m * 4,       [m * 4, m * 4])
        self.fp2 = FeaturePropagation(m * 4 + m * 2,       [m * 2, m * 2])
        self.fp1 = FeaturePropagation(m * 2 + in_channels, [m * 2, d_model])

        # ── Transformer decoder ──
        self.sa3_proj    = nn.Linear(m * 8, d_model)
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.decoder     = nn.ModuleList([
            _DecoderLayer(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_decoder_layers)
        ])

        # ── Heads ──
        self.cls_head = nn.Linear(d_model, num_classes)
        # mask: dot-product (queries · point_feats), scaled by 1/sqrt(d_model)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> dict:
        """
        features: [B, 9, N]
        returns : {'pred_logits': [B,Q,2],  'pred_masks': [B,Q,N]}
        """
        B, _, N = features.shape
        xyz = features[:, :3, :]

        # ── Backbone ──
        xyz1, f1 = self.sa1(xyz, features)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)

        f2_up = self.fp3(xyz2, xyz3, f2, f3)
        f1_up = self.fp2(xyz1, xyz2, f1, f2_up)
        f0    = self.fp1(xyz, xyz1, features, f1_up)          # [B, d_model, N]

        # ── Transformer memory (SA3 bottleneck) ──
        memory  = self.sa3_proj(f3.permute(0, 2, 1))          # [B, n3, d_model]

        # ── Decoder ──
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1).clone()
        for layer in self.decoder:
            queries = layer(queries, memory)                   # [B, Q, d_model]

        # ── Heads ──
        pred_logits = self.cls_head(queries)                   # [B, Q, 2]
        pred_masks  = torch.bmm(queries, f0) / (self._d_model ** 0.5)  # [B, Q, N]

        return {'pred_logits': pred_logits, 'pred_masks': pred_masks}


# ═══════════════════════════════════════════════════════════════
# Public interface  (evaluate.py 에서 호출)
# ═══════════════════════════════════════════════════════════════

def initialize_model(
    ckpt_path: str,
    device: torch.device,
    in_channels: int = 9,
    num_classes: int = 2,
) -> nn.Module:
    """
    체크포인트를 로드하고 eval 모드 모델 반환.
    train.py 가 저장하는 형식: {'model_state_dict': ..., 'epoch': ..., ...}
    """
    model = MaskTransformerNet(in_channels=in_channels, num_classes=num_classes).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state = (checkpoint.get('model_state_dict')
                 or checkpoint.get('state_dict')
                 or checkpoint.get('model')
                 or checkpoint)
    else:
        state = checkpoint

    # DataParallel prefix 제거
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run_inference(
    model: nn.Module,
    features: torch.Tensor,          # [B, 9, N]
    score_thr: float = 0.5,          # query fg-score threshold
    mask_thr:  float = 0.5,          # sigmoid(mask_logit) threshold
    min_pts:   int   = 50,           # minimum points per instance
    **kwargs,
) -> torch.Tensor:
    """
    Returns per-point instance labels [B, N]  (0=bg, 1~100=instance).

    Pipeline:
      1. forward → pred_logits [B,Q,2],  pred_masks [B,Q,N]
      2. fg_prob = softmax(pred_logits)[:,1]        — per-query fg score
      3. sort queries by fg_prob (descending)
      4. for each query above score_thr:
           binary_mask = sigmoid(pred_masks[q]) > mask_thr
           remove already-assigned points
           if remaining points >= min_pts → assign new instance id
    """
    B, _, N = features.shape
    device  = features.device

    with torch.no_grad():
        out = model(features)

    pred_logits = out['pred_logits']                        # [B, Q, 2]
    pred_masks  = out['pred_masks']                         # [B, Q, N]

    instance_labels = torch.zeros(B, N, dtype=torch.long, device=device)

    for b in range(B):
        fg_scores = torch.softmax(pred_logits[b], dim=-1)[:, 1]   # [Q]
        order     = fg_scores.argsort(descending=True)

        assigned  = torch.zeros(N, dtype=torch.bool, device=device)
        inst_id   = 1

        for q in order:
            if fg_scores[q].item() < score_thr or inst_id > 100:
                break
            mask = torch.sigmoid(pred_masks[b, q]) > mask_thr     # [N] bool
            mask = mask & ~assigned
            if mask.sum().item() < min_pts:
                continue
            instance_labels[b][mask] = inst_id
            assigned |= mask
            inst_id  += 1

    return instance_labels
