"""
fewshot.py
----------
Entry-point for few-shot probing, matching the original Point-BERT /
PointMamba protocol:

  - One fold per invocation (--way, --shot, --fold as CLI args)
  - Full encoder fine-tuning (not linear probing)
  - Mean ± std over 10 folds computed by the calling shell script

Usage (single fold):
    python fewshot.py \\
        --config cfgs/fewshot.yaml \\
        --ckpts  experiments/pretrain_modelnet/ckpt-best.pth \\
        --exp_name fewshot_5way10shot_fold0 \\
        --way 5 --shot 10 --fold 0

Usage (all folds via shell loop):
    for fold in $(seq 0 9); do
        python fewshot.py \\
            --config cfgs/fewshot.yaml \\
            --ckpts  experiments/pretrain_modelnet/ckpt-best.pth \\
            --exp_name fewshot_5way10shot \\
            --way 5 --shot 10 --fold $fold
    done

Prerequisites:
    Run generate_few_shot_data.py once before launching this script.
"""

import argparse
import os
import torch
import torch.distributed as dist
import yaml

from utils.config import get_config
from utils.logger import get_logger, print_log
from tools import builder
from tools.runner_fewshot import run_net


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='Few-shot fine-tuning with a pretrained point-cloud encoder')

    # Required
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config (e.g. cfgs/fewshot.yaml)')
    parser.add_argument(
        '--ckpts', type=str, required=True,
        help='Path to pretrained checkpoint (e.g. experiments/.../ckpt-best.pth)')

    # Few-shot episode definition — one fold per run
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from existing config in experiment_path')
    parser.add_argument('--way',  type=int, required=True,
                        help='N-way classification (e.g. 5 or 10)')
    parser.add_argument('--shot', type=int, required=True,
                        help='K-shot support samples per class (e.g. 10 or 20)')
    parser.add_argument('--fold', type=int, required=True,
                        help='Episode index 0-9 (averaged externally)')

    # Experiment
    parser.add_argument(
        '--exp_name', type=str, default='fewshot',
        help='Experiment name; logs saved to experiment_path/<exp_name>')
    parser.add_argument(
        '--experiment_path', type=str, default='experiments',
        help='Root directory for experiment outputs')

    # Misc
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--local_rank',  type=int, default=0)
    parser.add_argument('--log_name',    type=str, default='fewshot')
    parser.add_argument(
        '--opts', nargs='*', default=[],
        help='Override config values e.g. --opts max_epoch=300')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Distributed setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = world_size > 1

    if args.distributed:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    args.use_gpu    = torch.cuda.is_available()
    args.world_size = world_size
    args.rank       = dist.get_rank() if args.distributed else 0


# ─────────────────────────────────────────────────────────────────────────────
# Experiment directory
# ─────────────────────────────────────────────────────────────────────────────

def setup_experiment(args):
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

    # 3. Reproducibility — offset seed by fold so each fold has different
    #    random augmentation while remaining reproducible
    torch.manual_seed(args.seed + args.fold)

    # 4. Config
    config = get_config(args)

    # 5. Apply any CLI overrides
    if args.opts:
        for opt in args.opts:
            key, val = opt.split('=', 1)
            try:
                val = yaml.safe_load(val)
            except Exception:
                pass
            keys = key.split('.')
            node = config
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = val

    # 6. Logger
    log_file = os.path.join(args.experiment_path, f'{args.log_name}.log')
    logger = get_logger(args.log_name, log_file=log_file)


    if args.rank == 0:
        print_log(f'Experiment path : {args.experiment_path}', logger=logger)
        print_log(f'Config          : {args.config}',          logger=logger)
        print_log(f'Checkpoint      : {args.ckpts}',           logger=logger)
        print_log(f'Way / Shot / Fold: {args.way} / {args.shot} / {args.fold}',
                  logger=logger)
        print_log(f'Distributed     : {args.distributed}',     logger=logger)
        print_log(f'GPU available   : {args.use_gpu}',         logger=logger)

    # 7. Build model and load pretrained encoder weights
    base_model = builder.model_builder(config.model)
    builder.load_pretrained(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model = base_model.cuda()

    if args.distributed:
        base_model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        base_model = torch.nn.DataParallel(base_model).cuda()
        print_log('Using Data parallel ...', logger=logger)

    # 8. Run single fold
    best_acc = run_net(args, config, base_model)

    if args.rank == 0:
        print_log(
            f'[{args.way}-way {args.shot}-shot | fold {args.fold}] '
            f'best_acc = {best_acc:.2f}%',
            logger=logger)

    # 9. Cleanup
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
