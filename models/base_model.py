import torch
import torch.nn as nn
from types import SimpleNamespace
from mambapy.mamba import Mamba, MambaConfig
from point_scan import PointScan


class MambaBlocksWrapper(nn.Module):
    def __init__(self, d_model, n_layers, d_state=16):
        super().__init__()
        cfg = MambaConfig(d_model=d_model, n_layers=n_layers, d_state=d_state)
        self.mamba = Mamba(cfg)

    def forward(self, x, pos):
        return self.mamba(x + pos)


config = SimpleNamespace(
    trans_dim    = 384,
    depth        = 12,
    cls_dim      = 40,
    group_size   = 32,
    num_group    = 64,
    encoder_dims = 384,
)


class PointMambaScanTestable(PointScan):
    def __init__(self, config):
        super().__init__(config)
        self.use_cls_token = True
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.trans_dim))
        self.cls_pos   = nn.Parameter(torch.zeros(1, 1, config.trans_dim))
        self.blocks    = MambaBlocksWrapper(config.trans_dim, config.depth)


model = PointMambaScanTestable(config).eval()

B, N = 2, 1024
pts  = torch.randn(B, N, 3)

with torch.no_grad():
    out = model(pts)

assert out.shape == (B, 2 * config.trans_dim), f"Bad shape: {out.shape}"
assert not torch.isnan(out).any()
assert not torch.isinf(out).any()
print(f"✅ Mamba forward pass OK — output shape: {out.shape}")