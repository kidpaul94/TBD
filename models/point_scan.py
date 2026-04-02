import math

import torch
import torch.nn as nn

from pytorch3d.ops import knn_points as knn
from pytorch3d.ops import sample_farthest_points as fps
from models.serialization import Point


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """PointNet-style patch encoder (unchanged)."""

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : (B, G, N, 3)
        returns      : (B, G, C)
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)

        feature        = self.first_conv(point_groups.transpose(2, 1))      # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]         # BG 256 1
        feature        = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)             # BG 512 n
        feature        = self.second_conv(feature)                           # BG C   n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]        # BG C

        return feature_global.reshape(bs, g, self.encoder_channel)


# ─────────────────────────────────────────────────────────────────────────────
# Group  (original FPS + KNN — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class Group(nn.Module):
    """Baseline grouping: Farthest Point Sampling + K-Nearest Neighbours."""

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group  = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """
        xyz    : (B, N, 3)
        returns: neighborhood (B, G, M, 3),  center (B, G, 3)
        """
        batch_size, num_points, _ = xyz.shape

        center, _  = fps(xyz, K=self.num_group)              # B G 3
        knn_result = knn(center, xyz, K=self.group_size)
        idx        = knn_result.idx                           # B G M

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size

        idx_base = (
            torch.arange(0, batch_size, device=xyz.device)
            .view(-1, 1, 1) * num_points
        )
        idx          = (idx + idx_base).view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)    # center-subtract

        return neighborhood, center


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveGroup  (SPC-style geometry-adaptive grouping)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveGroup(nn.Module):
    """
    Geometry-adaptive patch tokenization via an implicit octree.

    Replaces FPS + KNN with a five-step pipeline:
      1. Normal estimation via PCA on a local KNN neighbourhood.
      2. Per-sample grid size (from a NN-distance percentile) and octree
         depth (from the bounding-box diagonal).  Both reuse the single KNN
         call from step 1 — no extra compute.
      3. Normal-variance computation per octree node via scatter-reduce.
      4. Greedy top-down token budget allocation: iteratively split the
         highest-variance node until exactly G leaf nodes are selected.
      5. Uniform sampling of M points per selected node followed by
         unit-sphere normalisation.

    Output contract is identical to Group:
        neighborhood : (B, G, M, 3)   centred, unit-sphere normalised
        center       : (B, G, 3)      node centroids in original coords

    Parameters
    ----------
    num_group  : int   G — number of tokens per point cloud.
    group_size : int   M — number of points sampled inside each token.
    k_normals  : int   K used for the shared normal-estimation KNN call.
    dist_pct   : float Percentile (0–100) of per-point NN distances used
                       as the grid-size estimate.  50 → median.
    min_pts    : int   Nodes with fewer than this many points are never
                       split and are marked ineligible for selection
                       (prevents degenerate tokens).
    """

    def __init__(
        self,
        num_group:  int,
        group_size: int,
        k_normals:  int   = 16,
        dist_pct:   float = 50.0,
        min_pts:    int   = 4,
    ):
        super().__init__()
        self.num_group  = num_group    # G
        self.group_size = group_size   # M
        self.k_normals  = k_normals
        self.dist_pct   = dist_pct
        self.min_pts    = min_pts

    # ── public forward ────────────────────────────────────────────────────

    def forward(self, xyz: torch.Tensor):
        """
        xyz    : (B, N, 3)
        returns: neighborhood (B, G, M, 3),  center (B, G, 3)
        """
        B = xyz.shape[0]

        # Step 1 — one KNN call, reused across steps 1 and 2
        normals, nn_dists = self._estimate_normals_and_distances(xyz)

        # Step 2 — grid quantisation (vectorised across B)
        grid_coord, depth_per_sample = self._build_grid(xyz, nn_dists)

        # Steps 3-5 — per-sample (octree structure varies per sample)
        neighborhoods, centers = [], []
        for b in range(B):
            nbhd_b, ctr_b = self._process_sample(
                xyz[b],
                normals[b],
                grid_coord[b],
                int(depth_per_sample[b].item()),
            )
            neighborhoods.append(nbhd_b)
            centers.append(ctr_b)

        neighborhood = torch.stack(neighborhoods, dim=0)   # (B, G, M, 3)
        center       = torch.stack(centers,       dim=0)   # (B, G, 3)
        return neighborhood, center

    # ── Step 1: normal estimation + NN distances ──────────────────────────

    def _estimate_normals_and_distances(self, xyz: torch.Tensor):
        """
        Single KNN call → normals (B, N, 3) and NN distances (B, N).

        Normals are the smallest eigenvector of each point's local
        covariance matrix (PCA).  Orientation is not fixed because only
        the variance (not direction) is needed downstream.
        """
        B, N, _ = xyz.shape
        K        = self.k_normals

        # knn_points returns squared distances; index 0 is the query point
        # itself (distance ≈ 0), so neighbours start at index 1.
        knn_out = knn(xyz, xyz, K=K + 1)          # dists/idx: (B, N, K+1)

        # ── NN distances (Step 2 input) ───────────────────────────────────
        nn_dists = torch.sqrt(
            knn_out.dists[:, :, 1].clamp(min=1e-8)
        )                                           # (B, N)

        # ── Gather neighbour coordinates ──────────────────────────────────
        nbr_idx  = knn_out.idx[:, :, 1:]           # (B, N, K)  skip self
        idx_base = (
            torch.arange(B, device=xyz.device).view(B, 1, 1) * N
        )
        flat_idx = (nbr_idx + idx_base).reshape(-1)             # (B*N*K,)
        xyz_flat = xyz.reshape(B * N, 3)
        neighbours = xyz_flat[flat_idx].reshape(B, N, K, 3)     # (B, N, K, 3)

        # ── Per-point covariance matrices ─────────────────────────────────
        centred  = neighbours - xyz.unsqueeze(2)                 # (B, N, K, 3)
        c_flat   = centred.reshape(B * N, K, 3)                  # (B*N, K, 3)
        cov      = torch.bmm(
            c_flat.transpose(1, 2), c_flat
        ) / K                                                    # (B*N, 3, 3)

        # ── PCA: smallest eigenvector = surface normal ────────────────────
        # torch.linalg.eigh returns eigenvalues in ascending order
        _, eigvecs = torch.linalg.eigh(cov)                      # (B*N, 3, 3)
        normals    = eigvecs[:, :, 0].reshape(B, N, 3)           # (B, N, 3)

        return normals, nn_dists

    # ── Step 2: grid quantisation ─────────────────────────────────────────

    def _build_grid(self, xyz: torch.Tensor, nn_dists: torch.Tensor):
        """
        Estimate a per-sample grid size from nn_dists and compute integer
        grid coordinates for every point.

        grid_size  : median (or chosen percentile) NN distance  →  (B,)
        depth      : ceil(log2(bbox_diag / grid_size))           →  (B,) int
        grid_coord : floor((xyz - xyz_min) / grid_size)          →  (B, N, 3) int64
        """
        B, N, _ = xyz.shape

        # Per-sample grid size —————————————————————————————————————————————
        # torch.quantile interpolates; clamp away zero to avoid log2(inf)
        grid_size = torch.quantile(
            nn_dists, self.dist_pct / 100.0, dim=1
        ).clamp(min=1e-6)                                        # (B,)

        # Per-sample octree depth ——————————————————————————————————————————
        xyz_min   = xyz.min(dim=1)[0]                            # (B, 3)
        xyz_max   = xyz.max(dim=1)[0]                            # (B, 3)
        bbox_diag = (xyz_max - xyz_min).norm(dim=-1)             # (B,)

        # Minimum depth so that 2^depth ≥ num_group  (enough leaf nodes)
        depth_min = max(1, math.ceil(math.log2(max(self.num_group + 1, 2))))

        raw_depth      = torch.log2(
            (bbox_diag / grid_size).clamp(min=1.0)
        )
        depth_per_sample = (
            raw_depth.ceil().long().clamp(min=depth_min, max=16)
        )                                                        # (B,)

        # Quantise all points (vectorised) ————————————————————————————————
        xyz_min_3d = xyz_min.unsqueeze(1)                        # (B, 1, 3)
        gs_3d      = grid_size.view(B, 1, 1)                     # (B, 1, 1)
        grid_coord = torch.floor(
            (xyz - xyz_min_3d) / gs_3d
        ).long()                                                 # (B, N, 3)

        return grid_coord, depth_per_sample

    # ── Steps 3-5: per-sample octree traversal and sampling ───────────────

    def _process_sample(
        self,
        xyz_b:        torch.Tensor,   # (N, 3)
        normals_b:    torch.Tensor,   # (N, 3)
        grid_coord_b: torch.Tensor,   # (N, 3) int64
        max_depth:    int,
    ):
        """
        Runs Steps 3 (variance), 4 (greedy budget allocation), and 5
        (sampling + normalisation) for a single point cloud.

        Returns
        -------
        neighborhood : (G, M, 3)
        center       : (G, 3)
        """
        device = xyz_b.device

        # Pre-compute node keys at every depth to avoid repeated shifts
        # keys_at[d] : (N, 3) int64 — node coordinate of each point at depth d
        keys_at = {
            d: grid_coord_b >> (max_depth - d)
            for d in range(1, max_depth + 1)
        }

        # ── helpers ───────────────────────────────────────────────────────

        def point_mask(d: int, ix: int, iy: int, iz: int) -> torch.Tensor:
            """Boolean mask (N,) of points inside node (d, ix, iy, iz)."""
            k = keys_at[d]
            return (k[:, 0] == ix) & (k[:, 1] == iy) & (k[:, 2] == iz)

        def node_variance(mask: torch.Tensor):
            """
            Normal variance and point count for nodes covered by mask.
            Returns (variance: float, count: int).
            Nodes with count < min_pts get variance=0 (ineligible).
            """
            count = int(mask.sum().item())
            if count < self.min_pts:
                return 0.0, count
            n_in  = normals_b[mask]                             # (count, 3)
            mean_n = n_in.mean(dim=0)                           # (3,)
            var    = ((n_in - mean_n) ** 2).sum(dim=-1).mean().item()
            return float(var), count

        # ── Step 3+4: greedy frontier initialisation ──────────────────────
        # Start at depth 1 — find occupied nodes among root's children
        unique_keys_1 = torch.unique(keys_at[1], dim=0)         # (K1, 3)

        frontier = []
        for k in range(unique_keys_1.shape[0]):
            ix, iy, iz = unique_keys_1[k].tolist()
            mask        = point_mask(1, ix, iy, iz)
            var, count  = node_variance(mask)
            frontier.append({
                'd':          1,
                'ix':         ix,
                'iy':         iy,
                'iz':         iz,
                'mask':       mask,
                'variance':   var,
                'count':      count,
                'splittable': (1 < max_depth) and (count >= self.min_pts),
            })

        # ── Step 4: greedy split until |frontier| == G ────────────────────
        while len(frontier) < self.num_group:
            # Collect nodes that can still be refined
            splittable = [
                (i, n) for i, n in enumerate(frontier) if n['splittable']
            ]
            if not splittable:
                break   # octree is fully expanded; exit early

            # Split the highest-variance eligible node
            splittable.sort(key=lambda x: x[1]['variance'], reverse=True)
            split_i, split_node = splittable[0]

            d  = split_node['d']
            ix = split_node['ix']
            iy = split_node['iy']
            iz = split_node['iz']

            # Enumerate up to 8 children at depth d+1
            children = []
            for dx in range(2):
                for dy in range(2):
                    for dz in range(2):
                        cix, ciy, ciz = ix * 2 + dx, iy * 2 + dy, iz * 2 + dz
                        if d + 1 > max_depth:
                            continue
                        mask = point_mask(d + 1, cix, ciy, ciz)
                        if not mask.any():
                            continue                 # empty voxel — skip
                        var, count = node_variance(mask)
                        children.append({
                            'd':          d + 1,
                            'ix':         cix,
                            'iy':         ciy,
                            'iz':         ciz,
                            'mask':       mask,
                            'variance':   var,
                            'count':      count,
                            'splittable': (d + 1 < max_depth)
                                          and (count >= self.min_pts),
                        })

            # Replace parent with its children in the frontier
            frontier.pop(split_i)
            frontier.extend(children)

        # ── Overshoot guard: merge lowest-variance nodes down to G ────────
        # Can occur when a single split adds several children and pushes
        # |frontier| beyond G.
        while len(frontier) > self.num_group:
            frontier.sort(key=lambda n: n['variance'])
            frontier.pop(0)                         # drop least informative

        # ── Undershoot guard: duplicate highest-variance node up to G ─────
        # Happens when the point cloud is nearly planar / has few occupied
        # voxels even at max depth.
        while len(frontier) < self.num_group:
            frontier.sort(key=lambda n: n['variance'], reverse=True)
            # Deep-copy only the dict (mask tensor is shared, read-only here)
            frontier.append(dict(frontier[0]))

        # ── Step 5: sample M points per node + unit-sphere normalise ──────
        neighborhoods, centers = [], []

        for node in frontier:
            mask  = node['mask']
            pts   = xyz_b[mask]                                  # (cnt, 3)
            cnt   = pts.shape[0]
            centroid = pts.mean(dim=0)                           # (3,)

            # Uniform sampling (with replacement when cnt < M)
            if cnt >= self.group_size:
                perm    = torch.randperm(cnt, device=device)[:self.group_size]
                sampled = pts[perm]                              # (M, 3)
            else:
                idx     = torch.randint(0, cnt, (self.group_size,),
                                        device=device)
                sampled = pts[idx]                               # (M, 3)

            # Centre then scale to unit sphere
            sampled = sampled - centroid
            scale   = sampled.norm(dim=-1).max().clamp(min=1e-8)
            sampled = sampled / scale                            # (M, 3)

            neighborhoods.append(sampled)
            centers.append(centroid)

        neighborhood = torch.stack(neighborhoods, dim=0)         # (G, M, 3)
        center       = torch.stack(centers,       dim=0)         # (G, 3)
        return neighborhood, center


# ─────────────────────────────────────────────────────────────────────────────
# Serialization helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def serialization(
    pos,
    feat=None,
    x_res=None,
    order="z",
    layers_outputs=None,
    grid_size=0.02,
):
    if layers_outputs is None:
        layers_outputs = []

    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord   = torch.floor(scaled_coord).to(torch.int64)
    min_coord    = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord   = grid_coord - min_coord

    batch_idx = (
        torch.arange(0, pos.shape[0], 1.0)
        .unsqueeze(1)
        .repeat(1, pos.shape[1])
        .to(torch.int64)
        .to(pos.device)
    )

    point_dict = Point(
        batch=batch_idx.flatten(),
        grid_coord=grid_coord.flatten(0, 1),
    )
    point_dict.serialization(order=order)

    order         = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if feat is not None:
        feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    for i in range(len(layers_outputs)):
        layers_outputs[i] = (
            layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
        )

    return pos, order, inverse_order, feat, x_res


def init_OrderScale(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta  = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=0.02)
    nn.init.normal_(beta,  std=0.02)
    return gamma, beta


def apply_OrderScale(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError(
            "Input tensor shape does not match the shape of the scale factor."
        )


def serialization_func(p, x, x_res, order, layers_outputs=None):
    if layers_outputs is None:
        layers_outputs = []
    p, order, inverse_order, x, x_res = serialization(
        p, x, x_res=x_res,
        order=order,
        layers_outputs=layers_outputs,
        grid_size=0.02,
    )
    return p, order, inverse_order, x, x_res


# ─────────────────────────────────────────────────────────────────────────────
# PointScan  (config-controlled grouping strategy)
# ─────────────────────────────────────────────────────────────────────────────

class PointScan(nn.Module):
    """
    Base feature-extraction module shared by all downstream models.

    group_mode (config field, default 'fps_knn')
    --------------------------------------------
    'fps_knn'  : original Group   — Farthest Point Sampling + KNN
    'adaptive' : AdaptiveGroup    — octree-based geometry-adaptive grouping

    Additional config fields for 'adaptive' mode (all optional):
        k_normals        int    KNN K for normal estimation        (default 16)
        dist_percentile  float  Percentile for grid-size estimate  (default 50)
        min_pts_per_node int    Min points per selectable node     (default  4)
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim    = config.trans_dim
        self.depth        = config.depth
        self.cls_dim      = config.cls_dim
        self.group_size   = config.group_size
        self.num_group    = config.num_group
        self.encoder_dims = config.encoder_dims

        # ── grouping strategy ─────────────────────────────────────────────
        self.group_mode = getattr(config, 'group_mode', 'fps_knn')

        if self.group_mode == 'adaptive':
            self.group_divider = AdaptiveGroup(
                num_group  = self.num_group,
                group_size = self.group_size,
                k_normals  = getattr(config, 'k_normals',         16),
                dist_pct   = getattr(config, 'dist_percentile',   50.0),
                min_pts    = getattr(config, 'min_pts_per_node',   4),
            )
        else:  # 'fps_knn'  (default — backward-compatible)
            self.group_divider = Group(
                num_group  = self.num_group,
                group_size = self.group_size,
            )

        # ── shared modules (identical for both grouping modes) ────────────
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(
            self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(
            self.trans_dim)

        self.serialization = getattr(config, 'serialization', 'unidirectional')

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens   = self.encoder(neighborhood)       # B G C
        pos                  = self.pos_embed(center)           # B G C

        if self.serialization == 'bidirectional':
            _, _, _, tokens_fwd, pos_fwd = serialization_func(
                center, group_input_tokens, pos, 'hilbert')
            _, _, _, tokens_bwd, pos_bwd = serialization_func(
                center, group_input_tokens, pos, 'hilbert-trans')
            tokens_fwd = apply_OrderScale(
                tokens_fwd, self.OrderScale_gamma_1, self.OrderScale_beta_1)
            tokens_bwd = apply_OrderScale(
                tokens_bwd, self.OrderScale_gamma_2, self.OrderScale_beta_2)

            cls_token = self.cls_token.expand(tokens_fwd.size(0), -1, -1)
            cls_pos   = self.cls_pos.expand(tokens_fwd.size(0), -1, -1)
            pos                = torch.cat([pos_fwd, pos_bwd, cls_token], dim=1)
            group_input_tokens = torch.cat(
                [tokens_fwd, tokens_bwd, cls_pos], dim=1)

        else:  # unidirectional (default)
            _, _, _, group_input_tokens, pos = serialization_func(
                center, group_input_tokens, pos, 'hilbert')
            group_input_tokens = apply_OrderScale(
                group_input_tokens,
                self.OrderScale_gamma_1, self.OrderScale_beta_1,
            )
            cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
            cls_pos   = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
            pos                = torch.cat([pos, cls_token], dim=1)
            group_input_tokens = torch.cat([group_input_tokens, cls_pos], dim=1)

        x = self.blocks(group_input_tokens, pos)

        if self.use_cls_token:
            cls_token = x[:, -1, :]
            concat_f  = torch.cat([cls_token, x.mean(1)], dim=1)

        return concat_f