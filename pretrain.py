"""
pretrain.py
-----------
Entry-point for pre-training. Parses CLI arguments, initialises the
experiment directory, loggers, and TensorBoard writers, then delegates
entirely to tools/runner_pretrain.py::run_net().

Usage (single GPU):
    python pretrain.py --config cfgs/pretrain_modelnet.yaml \
                       --exp_name pretrain_modelnet

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=4 pretrain.py \
        --config cfgs/pretrain_modelnet.yaml \
        --exp_name pretrain_modelnet

Resume:
    python pretrain.py --config cfgs/pretrain_modelnet.yaml \
                       --exp_name pretrain_modelnet \
                       --resume
"""

import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.config import get_config
from utils.logger import get_logger, print_log
from tools.runner_pretrain import run_net


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Point-cloud MAE pre-training")

    # Config
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config (e.g. cfgs/pretrain_modelnet.yaml)')

    # Experiment
    parser.add_argument(
        '--exp_name', type=str, default='pretrain',
        help='Experiment name; checkpoints saved to experiment_path/<exp_name>')
    parser.add_argument(
    '--opts', nargs='*', default=[],
    help="Override config values, e.g. --opts max_epoch=1 total_bs=4")
    parser.add_argument(
        '--experiment_path', type=str, default='experiments',
        help='Root directory for all experiment outputs')

    # Training control
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Resume from the last checkpoint in experiment_path')
    parser.add_argument(
        '--start_ckpts', type=str, default=None,
        help='Load weights from this checkpoint (no resume, just init)')
    parser.add_argument(
        '--val_freq', type=int, default=10,
        help='Run SVM validation every N epochs (requires dataset.extra_train)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for reproducibility')

    # Distributed / GPU
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader worker processes per GPU')
    parser.add_argument(
        '--local_rank', type=int, default=0,
        help='(Set automatically by torchrun) Local GPU rank')
    parser.add_argument(
        '--sync_bn', action='store_true', default=False,
        help='Convert all BN layers to SyncBN in DDP mode')

    # Logging
    parser.add_argument(
        '--log_name', type=str, default='pretrain',
        help='Logger name (used by utils/logger.py)')
    parser.add_argument(
        '--tensorboard', action='store_true', default=False,
        help='Enable TensorBoard logging')

    args = parser.parse_args()
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Distributed setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed(args):
    """Detect torchrun environment and initialise process group."""
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = world_size > 1

    if args.distributed:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    args.use_gpu = torch.cuda.is_available()
    args.world_size = world_size
    args.rank = dist.get_rank() if args.distributed else 0


# ─────────────────────────────────────────────────────────────────────────────
# Experiment directory
# ─────────────────────────────────────────────────────────────────────────────

def setup_experiment(args):
    """
    Build args.experiment_path = <experiment_path>/<exp_name>.
    This path is used by:
      - get_config() to find / save config.yaml when resuming
      - builder.save_checkpoint() for checkpoint files
      - TensorBoard writers
    """
    args.experiment_path = os.path.join(args.experiment_path, args.exp_name)
    if args.rank == 0:
        os.makedirs(args.experiment_path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # 1. Distributed init
    setup_distributed(args)

    # 2. Experiment directory
    setup_experiment(args)

    # 3. Reproducibility
    torch.manual_seed(args.seed + args.rank)

    # 4. Config  (uses existing utils/config.py::get_config)
    #    - If args.resume is True, get_config loads from experiment_path/config.yaml
    #    - Otherwise loads from args.config and copies it to experiment_path/
    config = get_config(args)

    if args.opts:
        for opt in args.opts:
            key, val = opt.split('=', 1)
            try:
                import yaml as _yaml
                val = _yaml.safe_load(val)
            except Exception:
                pass
            keys = key.split('.')
            node = config
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = val

    # 5. Logger
    logger = get_logger(args.log_name)
    if args.rank == 0:
        print_log(f'Experiment path : {args.experiment_path}', logger=logger)
        print_log(f'Config          : {args.config}',          logger=logger)
        print_log(f'Distributed     : {args.distributed}',     logger=logger)
        print_log(f'GPU available   : {args.use_gpu}',         logger=logger)

    # 6. TensorBoard writers (rank-0 only)
    train_writer = val_writer = None
    if args.tensorboard and args.rank == 0:
        tb_dir = os.path.join(args.experiment_path, 'tensorboard')
        train_writer = SummaryWriter(log_dir=os.path.join(tb_dir, 'train'))
        val_writer   = SummaryWriter(log_dir=os.path.join(tb_dir, 'val'))
        print_log(f'TensorBoard logs: {tb_dir}', logger=logger)

    # 7. Run
    run_net(args, config, train_writer=train_writer, val_writer=val_writer)

    # 8. Cleanup
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
