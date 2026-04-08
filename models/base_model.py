"""
models/base_model.py
--------------------
Masked Auto-Encoder (MAE) with a Mamba SSM backbone and
bidirectional Hilbert-curve serialization.

Dependencies
------------
  mambapy   : pure-Python Mamba (no CUDA extension required)
  pytorch3d : fps / knn (used by PointScan / Group)

Modes
-----
  Training   (noaug=False) : random patch masking → Mamba encoder →
                              Mamba decoder → Chamfer loss (scalar)
  Evaluation (noaug=True)  : full bidirectional Hilbert scan →
                              cls + mean global feature [B, 2*trans_dim]
                              (delegates entirely to PointScan.forward)
"""

import torch
import torch.nn as nn
from types import SimpleNamespace
from timm.models.layers import trunc_normal_
from mambapy.mamba import Mamba, MambaConfig
from pytorch3d.loss import chamfer_distance

from models.build import MODELS
from models.point_scan import PointScan, apply_OrderScale, serialization_func


# ─────────────────────────────────────────────────────────────────────────────
# Mamba wrappers
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlocksWrapper(nn.Module):
    """
    Thin wrapper around mambapy.Mamba that accepts an additive positional
    embedding, matching the (x, pos) signature used in PointScan.forward().

    forward: x, pos [B, L, D] -> Mamba(x + pos) [B, L, D]
    """

    def __init__(self, d_model: int, n_layers: int, d_state: int = 16):
        super().__init__()
        cfg = MambaConfig(d_model=d_model, n_layers=n_layers, d_state=d_state)
        self.mamba = Mamba(cfg)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.mamba(x + pos)


class MambaDecoder(nn.Module):
    """
    Decoder Mamba stack.  Receives tokens with positional embeddings
    already added, so it only needs a single tensor input.

    forward: x [B, L, D] -> Mamba(x) [B, L, D]
    """

    def __init__(self, d_model: int, n_layers: int, d_state: int = 16):
        super().__init__()
        cfg = MambaConfig(d_model=d_model, n_layers=n_layers, d_state=d_state)
        self.mamba = Mamba(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

@MODELS.register_module()
class Point_MAE_Mamba_serializationV2(PointScan):
    """
    MAE pre-training model with Mamba SSM backbone and Hilbert serialization.

    Instantiate via YAML config::

        from models.build import build_model_from_cfg
        from utils.config import cfg_from_yaml_file
        cfg = cfg_from_yaml_file('cfgs/pretrain_modelnet.yaml')
        model = build_model_from_cfg(cfg.model)

    Or directly::

        model = Point_MAE_Mamba_serializationV2.build_from_cfg(cfg.model)

    Config schema (``model`` section of YAML)
    -----------------------------------------
    Baseline (fps_knn) — backward-compatible, no new keys required::

        NAME:       Point_MAE_Mamba_serializationV2
        group_size: 32
        num_group:  64
        loss:       cdl2          # 'cdl1' | 'cdl2'
        mamba_config:
          mask_ratio:    0.6
          mask_type:     rand
          trans_dim:     384
          encoder_dims:  384
          depth:         12
          decoder_depth: 4
          serialization: unidirectional   # 'unidirectional' | 'bidirectional'
        drop_path: 0.1
        drop_out:  0.0
    """

    def __init__(self, config):
        super().__init__(config)

        # ── cls token (feature-extraction mode only) ──────────────────────
        self.use_cls_token = True
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.trans_dim))
        self.cls_pos   = nn.Parameter(torch.zeros(1, 1, config.trans_dim))

        # ── encoder (shared with feature-extraction path in PointScan) ────
        self.blocks = MambaBlocksWrapper(config.trans_dim, config.depth)

        # ── MAE learnable mask token ───────────────────────────────────────
        self.mask_ratio = config.mask_ratio
        self.mask_type  = config.mask_type
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.trans_dim))

        # ── decoder ───────────────────────────────────────────────────────
        self.decoder      = MambaDecoder(config.trans_dim, config.decoder_depth)
        self.decoder_norm = nn.LayerNorm(config.trans_dim)
        self.decoder_head = nn.Linear(config.trans_dim, 3)   # predict xyz

        # ── loss type ─────────────────────────────────────────────────────
        self._loss_norm = 1 if getattr(config, 'loss', 'cdl2') == 'cdl1' else 2
        self.serialization = getattr(config, 'serialization', 'unidirectional')

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────────

    def _init_weights(self):
        trunc_normal_(self.cls_token,  std=0.02)
        trunc_normal_(self.cls_pos,    std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        nn.init.zeros_(self.decoder_head.bias)

    # ── masking ───────────────────────────────────────────────────────────

    def _mask(self, tokens: torch.Tensor, pos: torch.Tensor):
        """
        Random masking of Hilbert-ordered patch tokens.

        Args:
            tokens  [B, G, D]
            pos     [B, G, D]

        Returns:
            vis_tokens  [B, G-M, D]
            vis_pos     [B, G-M, D]
            mask_idx    [B, M]        indices of masked patches
            vis_idx     [B, G-M]      indices of visible patches
        """
        B, G, D  = tokens.shape
        num_mask = int(G * self.mask_ratio)

        noise    = torch.rand(B, G, device=tokens.device)
        ids      = noise.argsort(dim=1)
        mask_idx = ids[:, :num_mask]
        vis_idx  = ids[:, num_mask:]

        def _gather(t, idx):
            return torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, D))

        return _gather(tokens, vis_idx), _gather(pos, vis_idx), mask_idx, vis_idx

    # ── Chamfer loss ──────────────────────────────────────────────────────

    def _chamfer_loss(self, pred: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        """Bidirectional Chamfer distance. pred / target: [B, M, 3]."""
        loss, _ = chamfer_distance(pred, target, norm=self._loss_norm)
        return loss

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, pts: torch.Tensor,
                noaug: bool = False) -> torch.Tensor:
        """
        Args:
            pts    [B, N, 3]  raw point cloud
            noaug  bool       True -> feature-extraction (SVM validation)

        Returns:
            loss      scalar           when noaug=False  (pre-training)
            concat_f  [B, 2*trans_dim] when noaug=True   (feature extraction)
        """
        if noaug:
            # Bidirectional Hilbert scan + cls + mean pooling.
            # Fully handled by PointScan.forward().
            return super().forward(pts)

        # ── 1. Grouping ───
        neighborhood, center = self.group_divider(pts)    # [B,G,M,3], [B,G,3]

        # ── 2. Encode patches ─────────────────────────────────────────────
        tokens = self.encoder(neighborhood)               # [B, G, D]
        pos    = self.pos_embed(center)                   # [B, G, D]

        # ── 3. Hilbert serialisation ──────────────────────────────────────
        #   'bidirectional': forward + backward scans concatenated [B, 2G, D]
        #                    matches the noaug=True feature-extraction path
        #   'unidirectional': forward scan only [B, G, D]
        #                     lower memory cost, faster training
        if self.serialization == 'bidirectional':
            center_fwd, _, _, tokens_fwd, pos_fwd = serialization_func(
                center, tokens, pos, 'hilbert')
            center_bwd, _, _, tokens_bwd, pos_bwd = serialization_func(
                center, tokens, pos, 'hilbert-trans')
            tokens_fwd = apply_OrderScale(
                tokens_fwd, self.OrderScale_gamma_1, self.OrderScale_beta_1)
            tokens_bwd = apply_OrderScale(
                tokens_bwd, self.OrderScale_gamma_2, self.OrderScale_beta_2)
            tokens        = torch.cat([tokens_fwd, tokens_bwd], dim=1)
            pos           = torch.cat([pos_fwd,    pos_bwd],    dim=1)
            center_sorted = torch.cat([center_fwd, center_bwd], dim=1)
        else:  # unidirectional
            center_sorted, _, _, tokens, pos = serialization_func(
                center, tokens, pos, 'hilbert')
            tokens = apply_OrderScale(
                tokens, self.OrderScale_gamma_1, self.OrderScale_beta_1)

        # ── 4. Random masking ─────────────────────────────────────────────
        vis_tokens, vis_pos, mask_idx, vis_idx = self._mask(tokens, pos)

        # ── 5. Encode visible patches ─────────────────────────────────────
        vis_encoded = self.blocks(vis_tokens, vis_pos)    # [B, G-M, D]

        # ── 6. Reconstruct full sequence (encoded visible + mask tokens) ──
        B, G, D  = tokens.shape
        num_mask = mask_idx.shape[1]

        def _scatter(src, idx):
            """Scatter src into a [B, G, D] zero tensor at positions idx."""
            return torch.zeros(B, G, D, device=pts.device).scatter_(
                1, idx.unsqueeze(-1).expand(-1, -1, D), src)

        full = (
            _scatter(vis_encoded, vis_idx)
            + _scatter(self.mask_token.expand(B, num_mask, -1), mask_idx)
        )
        # Add positional embedding before decoding so masked tokens know
        # their spatial location.
        full = full + pos

        # ── 7. Decode ─────────────────────────────────────────────────────
        decoded  = self.decoder(full)                     # [B, G, D]
        decoded  = self.decoder_norm(decoded)
        pred_xyz = self.decoder_head(decoded)             # [B, G, 3]

        # Extract predictions and targets at masked positions only
        def _gather3(t, idx):
            return torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        pred_masked   = _gather3(pred_xyz,      mask_idx)  # [B, M, 3]
        target_masked = _gather3(center_sorted, mask_idx)  # [B, M, 3]

        # ── 8. Chamfer loss ───────────────────────────────────────────────
        return self._chamfer_loss(pred_masked, target_masked)

    # ── YAML config factory ───────────────────────────────────────────────

    @classmethod
    def build_from_cfg(cls,
                       cfg) -> "Point_MAE_Mamba_serializationV2":
        """
        Build from the EasyDict ``model`` section loaded by utils/config.py.

        Flattens the nested ``mamba_config`` block into a SimpleNamespace
        that matches the flat attribute access expected by PointScan.__init__.

        Args:
            cfg: EasyDict with the structure shown in the class docstring.

        Returns:
            Instantiated Point_MAE_Mamba_serializationV2.
        """
        mc = cfg.get('mamba_config', {})

        config = SimpleNamespace(
            # ── PointScan / shared fields ─────────────────────────────────
            serialization = mc.get('serialization',  'unidirectional'),
            trans_dim     = mc.get('trans_dim',       384),
            depth         = mc.get('depth',           12),
            encoder_dims  = mc.get('encoder_dims',    384),
            group_size    = cfg.get('group_size',     32),
            num_group     = cfg.get('num_group',      64),
            cls_dim       = cfg.get('cls_dim',        0),   # unused in pretraining

            # ── MAE-specific fields ───────────────────────────────────────
            mask_ratio    = mc.get('mask_ratio',    0.6),
            mask_type     = mc.get('mask_type',     'rand'),
            decoder_depth = mc.get('decoder_depth', 4),
            loss          = cfg.get('loss',         'cdl2'),
        )
        return cls(config)