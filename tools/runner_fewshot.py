import torch
import torch.nn as nn
import numpy as np
from tools import builder
from pytorch3d.ops import sample_farthest_points as fps
from utils.logger import *
from utils.AverageMeter import AverageMeter
from timm.scheduler import CosineLRScheduler
from torchvision import transforms
from datasets import data_transforms

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Classification head
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, n_way: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_way)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# Runner — single (way, shot, fold) invocation
# ─────────────────────────────────────────────────────────────────────────────

def run_net(args, config, base_model):
    """
    Fine-tune the pretrained encoder + a fresh classification head on a
    single few-shot episode defined by (way, shot, fold).

    Mirrors the original Point-BERT / PointMamba protocol:
      - Full encoder fine-tuning (not linear probing)
      - One fold per invocation; mean ± std computed across 10 runs
        by the calling shell script

    Args:
        args       : parsed args with .way, .shot, .fold, .local_rank, .use_gpu
        config     : EasyDict from fewshot.yaml
        base_model : encoder with pretrained weights loaded (DataParallel / DDP)

    Returns:
        best_acc (float) : best query-set accuracy (%) over training epochs
    """
    logger   = get_logger(args.log_name)
    device   = (torch.device('cuda', args.local_rank)
                if args.use_gpu else torch.device('cpu'))

    way      = args.way
    shot     = args.shot
    fold     = args.fold
    npoints  = config.dataset.train.others.get('npoints', 1024)
    feat_dim = 2 * config.model.mamba_config.trans_dim
    max_epoch = config.max_epoch

    print_log('=' * 60, logger=logger)
    print_log(f'Few-shot fine-tuning: {way}-way  {shot}-shot  fold {fold}',
              logger=logger)
    print_log(f'  feat_dim  : {feat_dim}', logger=logger)
    print_log(f'  npoints   : {npoints}',  logger=logger)
    print_log(f'  max_epoch : {max_epoch}', logger=logger)
    print_log('=' * 60, logger=logger)

    # ── inject way / shot / fold into dataset others ──────────────────────
    for split in ('train', 'val'):
        others       = config.dataset[split].others
        others.way   = way
        others.shot  = shot
        others.fold  = fold

    _, train_loader = builder.dataset_builder(
        args, config.dataset.train, config.total_bs)
    _, test_loader  = builder.dataset_builder(
        args, config.dataset.val,   config.total_bs)

    # ── classification head ───────────────────────────────────────────────
    cls_head = ClassificationHead(feat_dim, way).to(device)

    # ── optimizer over encoder + head ─────────────────────────────────────
    # base_model is DataParallel / DDP wrapped; access params directly
    all_params = list(base_model.parameters()) + list(cls_head.parameters())
    optimizer  = torch.optim.AdamW(
        all_params,
        lr=config.optimizer.kwargs.lr,
        weight_decay=config.optimizer.kwargs.weight_decay)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max_epoch,
        lr_min=1e-6,
        cycle_decay=0.1,
        warmup_lr_init=1e-6,
        warmup_t=config.scheduler.kwargs.initial_epochs,
        cycle_limit=1,
        t_in_epochs=True)

    criterion  = nn.CrossEntropyLoss()
    grad_clip  = config.get('grad_norm_clip', 10)
    best_acc   = 0.0

    for epoch in range(max_epoch):

        # ── train ─────────────────────────────────────────────────────────
        base_model.train()
        cls_head.train()
        losses = AverageMeter(['Loss'])

        for _, _, data in train_loader:
            points = data[0].to(device)
            labels = data[1].to(device)

            points, _ = fps(points, K=npoints)
            points     = train_transforms(points)

            feat   = base_model(points, noaug=True)
            logits = cls_head(feat)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, grad_clip)
            optimizer.step()

            losses.update([loss.item()])

        scheduler.step(epoch)

        # ── evaluate every 10 epochs and on the final epoch ───────────────
        if epoch % 10 == 0 or epoch == max_epoch - 1:
            base_model.eval()
            cls_head.eval()
            correct = total = 0

            with torch.no_grad():
                for _, _, data in test_loader:
                    points = data[0].to(device)
                    labels = data[1].to(device)
                    points, _ = fps(points, K=npoints)
                    feat   = base_model(points, noaug=True)
                    pred   = cls_head(feat).argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total   += labels.size(0)

            acc = correct / total * 100.
            if acc > best_acc:
                best_acc = acc

            print_log(
                f'[{way}-way {shot}-shot | fold {fold}] '
                f'Epoch {epoch:3d}/{max_epoch}  '
                f'loss={losses.avg(0):.4f}  '
                f'acc={acc:.2f}%  best={best_acc:.2f}%',
                logger=logger)

    print_log(
        f'[{way}-way {shot}-shot | fold {fold}] '
        f'Final best acc = {best_acc:.2f}%',
        logger=logger)

    return best_acc
