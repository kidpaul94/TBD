import torch
import torch.nn as nn
from tools import builder
from pytorch3d.ops import sample_farthest_points as fps
from utils import dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        # data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    # ── build dataset ─────────────────────────────────────────────────────
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train, config.total_bs), \
        builder.dataset_builder(args, config.dataset.val, config.total_bs)

    # extra_train_dataloader is optional: used for SVM validation only.
    # If the config does not define dataset.extra_train, validation is skipped.
    (_, extra_train_dataloader) = \
        builder.dataset_builder(args, config.dataset.extra_train, config.total_bs) \
        if config.dataset.get('extra_train') else (None, None)

    # ── build model ───────────────────────────────────────────────────────
    base_model = builder.model_builder(config.model)
    print_log(f"Model: {base_model}", logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    # ── parameter setup ───────────────────────────────────────────────────
    start_epoch = 0
    best_metrics = Acc_Metric(float('inf'))
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args,
                                                         logger=logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # ── DDP ───────────────────────────────────────────────────────────────
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # ── optimizer & scheduler ─────────────────────────────────────────────
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # ── training ──────────────────────────────────────────────────────────
    base_model.zero_grad()

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        base_model.train()
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            # config.npoints (top-level YAML field, e.g. 1024) is the target
            # point count fed to the model. _base_.N_POINTS is the size
            # stored in the .dat file (8192) — only used to locate that file.
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME

            if dataset_name == 'ShapeNet' or dataset_name == 'UnlabeledHybrid':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points, _ = fps(points, K=npoints)
            else:
                raise NotImplementedError(
                    f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            points = train_transforms(points)

            loss = base_model(points)

            loss = loss.mean()
            loss.backward()

            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item() * 1000])
            else:
                losses.update([loss.item() * 1000])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR',
                                        optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log(
                    '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) '
                    'DataTime = %.3f (s) Losses = %s lr = %.6f' %
                    (epoch, config.max_epoch, idx + 1, n_batches,
                     batch_time.val(), data_time.val(),
                     ['%.4f' % l for l in losses.val()],
                     optimizer.param_groups[0]['lr']),
                    logger=logger)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)

        print_log(
            '[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch, epoch_end_time - epoch_start_time,
             ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']),
            logger=logger)

        # ── Reconstruction validation ──────────────────────────────────────
        # Runs every val_freq epochs on test_dataloader using the MAE
        # forward path (mask → encode → decode → Chamfer loss).
        val_freq = config.get('val_freq', 10)
        if epoch % val_freq == 0 and epoch != 0:
            val_loss = validate_reconstruction(base_model, test_dataloader,
                                               epoch, val_writer, args,
                                               config, logger=logger)
            if val_loss < best_metrics.acc:
                best_metrics = Acc_Metric(val_loss)
                builder.save_checkpoint(base_model, optimizer, epoch,
                                        metrics, best_metrics, 'ckpt-best',
                                        args, logger=logger)

        # ── SVM validation (optional) ─────────────────────────────────────
        # Runs only if dataset.extra_train is defined in the config.
        if (extra_train_dataloader is not None
                and epoch % val_freq == 0 and epoch != 0):
            metrics = validate(base_model, extra_train_dataloader,
                               test_dataloader, epoch, val_writer,
                               args, config, logger=logger)

        # ── Checkpoint saving ─────────────────────────────────────────────
        builder.save_checkpoint(base_model, optimizer, epoch, metrics,
                                best_metrics, 'ckpt-last', args,
                                logger=logger)
        if epoch % 25 == 0 and epoch >= 250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics,
                                    best_metrics, f'ckpt-epoch-{epoch:03d}',
                                    args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate_reconstruction(base_model, test_dataloader, epoch, val_writer,
                            args, config, logger=None):
    print_log(f"[VALIDATION] Reconstruction loss @ epoch {epoch}", logger=logger)
    base_model.eval()

    losses = AverageMeter(['Loss'])
    npoints = config.dataset.val.others.npoints
    dataset_name = config.dataset.val._base_.NAME

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if dataset_name == 'ShapeNet' or dataset_name == 'UnlabeledHybrid':
                points = data.cuda()
            else:  # ModelNet
                points = data[0].cuda()
            points, _ = fps(points, K=npoints)
            assert points.size(1) == npoints

            loss = base_model(points)
            try:
                loss = loss.mean()
            except:
                pass
            losses.update([loss.item() * 1000])

    print_log(f'[Validation] EPOCH: {epoch}  Reconstruction Loss = {losses.avg(0):.4f}',
              logger=logger)

    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Reconstruction', losses.avg(0), epoch)

    return losses.avg(0)


def validate(base_model, extra_train_dataloader, test_dataloader, epoch,
             val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()

    test_features = []
    test_label = []
    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(
                extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            points, _ = fps(points, K=npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            train_features.append(feature.detach())
            train_label.append(label.view(-1).detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(
                test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            points, _ = fps(points, K=npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            test_features.append(feature.detach())
            test_label.append(label.view(-1).detach())

    train_features = torch.cat(train_features, dim=0)
    train_label = torch.cat(train_label, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_label = torch.cat(test_label, dim=0)

    if args.distributed:
        train_features = dist_utils.gather_tensor(train_features, args)
        train_label = dist_utils.gather_tensor(train_label, args)
        test_features = dist_utils.gather_tensor(test_features, args)
        test_label = dist_utils.gather_tensor(test_label, args)

    svm_acc = evaluate_svm(
        train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
        test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

    print_log('[Validation] EPOCH: %d acc = %.4f' % (epoch, svm_acc),
              logger=logger)

    if args.distributed:
        torch.cuda.synchronize()

    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass