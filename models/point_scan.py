import torch
import torch.nn as nn

from pytorch3d.ops import knn_points as knn
from pytorch3d.ops import sample_farthest_points as fps
from serialization import Point


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        center, _ = fps(xyz, K=self.num_group)  # B G 3
        knn_result = knn(center, xyz, K=self.group_size)
        idx = knn_result.idx  # B G M

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


def serialization(pos, feat=None, x_res=None, order="z", layers_outputs=[], grid_size=0.02):
    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if feat is not None:
        feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    for i in range(len(layers_outputs)):
        layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    return pos, order, inverse_order, feat, x_res


def init_OrderScale(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta


def apply_OrderScale(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


def serialization_func(p, x, x_res, order, layers_outputs=[]):
    p, order, inverse_order, x, x_res = serialization(p, x, x_res=x_res, order=order,
                                                      layers_outputs=layers_outputs,
                                                      grid_size=0.02)
    return p, order, inverse_order, x, x_res


class PointScan(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointScan, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(self.trans_dim)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)  # B G C

        # reordering strategy
        _, _, _, group_input_tokens_forward, pos_forward = serialization_func(center, group_input_tokens, pos,
                                                                              'hilbert')
        _, _, _, group_input_tokens_backward, pos_backward = serialization_func(center, group_input_tokens, pos,
                                                                                'hilbert-trans')
        group_input_tokens_forward = apply_OrderScale(group_input_tokens_forward,
                                                      self.OrderScale_gamma_1, self.OrderScale_beta_1)
        group_input_tokens_backward = apply_OrderScale(group_input_tokens_backward,
                                                       self.OrderScale_gamma_2, self.OrderScale_beta_2)
        cls_token = self.cls_token.expand(group_input_tokens_forward.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens_forward.size(0), -1, -1)
        pos = torch.cat([pos_forward, pos_backward, cls_token], dim=1)
        group_input_tokens = torch.cat([group_input_tokens_forward, group_input_tokens_backward, cls_pos], dim=1)

        x = group_input_tokens
        x = self.blocks(x, pos)
        if self.use_cls_token:
            cls_token = x[:, -1, :]
            concat_f = torch.cat([cls_token, x.mean(1)], dim=1)

        return concat_f
