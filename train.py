import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.scheduler import CosineAnnealingWarmUpSingle, CosineAnnealingWarmUpRestarts
# import torchvision.models as models
import model as models
from utils.transforms import transforms_train, transforms_test, NormalizePerImage
import math
from utils.losses import *
from utils import LARC


parser = argparse.ArgumentParser(description='PyTorch FER Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--data', default='data', type=str, help='directory that includes train, (validation), and test folder.')
parser.add_argument('--model-version', metavar='ARCH', default='mobilenet_v3_small',
                    choices=['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet50'],
                    help='model architecture')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
#parser.add_argument('-b', '--batch-size', default=32, type=int,
#                    metavar='N',
#                    help='mini-batch size (default: 32), this is the total '
#                         'batch size of all GPUs on the current node when '
#                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--batch-loss', metavar='LOSS_TYPE', type= str, default='FL',
                    choices=['CE', 'FL'])
parser.add_argument('--if-momentum-scheduler', action='store_true', help='If use momentum scheduler')
parser.add_argument('--scheduler', default='single', choices=[None, 'single', 'multi'], help='scheduler type. None for no scheduler.')
parser.add_argument('--num-categories', default=7, type=int, help='number of categories (default: 7)')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--identifier', default='ex', help='identifier to be used.')
best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.model_version))
    if 'eff' in args.model_version:
        model = models.__dict__[args.model_version](num_classes = args.num_categories)
    else:
        model = models.__dict__[args.model_version](num_classes = args.num_categories, pretrained=True)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.model_version.startswith('alexnet') or args.model_version.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if args.batch_loss == 'CE':
        criterion = CrossEntropyLoss().cuda(args.gpu)
    elif args.batch_loss == 'FL':
        criterion = FocalLoss().cuda(args.gpu)

    _optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer = LARC(_optimizer)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            _optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')
    # val_dataset.transform = transforms_test()

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms_train())

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    random.seed(args.seed)
    random.shuffle(train_dataset.samples)

    if ('val' not in os.listdir(args.data)) and ('valication' not in os.listdir(args.data)):
        # training set의 90%를 training set으로 사용하고 나머지 10%는 validation set으로 사용.
        num_val = int(0.1*len(train_dataset))
        num_tr = int(len(train_dataset) - num_val)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_tr, num_val], generator=torch.Generator().manual_seed(args.seed))

    else:
        if 'val' in os.listdir(args.data):
            valdir = os.path.join(args.data, 'val')
        elif 'valication' in os.listdir(args.data):
            valdir = os.path.join(args.data, 'val')
        val_dataset = train_dataset = datasets.ImageFolder(valdir,
        transforms_test())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms_test()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False)

    if args.scheduler == 'multi':
        scheduler = CosineAnnealingWarmUpRestarts(_optimizer, eta_max=args.lr * math.sqrt(args.batch_size),
                                                  step_total=len(train_loader) * args.epochs)
    #            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, len(loader_tr_cv) * (args.epoch_steps[-1] - args.epoch_steps[0]), eta_min=0)
    elif args.scheduler == 'single':
        scheduler = CosineAnnealingWarmUpSingle(_optimizer, max_lr=args.lr * math.sqrt(args.batch_size),
                                                epochs=args.epochs, steps_per_epoch=len(train_loader),
                                                div_factor=math.sqrt(args.batch_size),
                                                cycle_momentum=args.if_momentum_scheduler)

# FP-16 의 mixed-precision 훈련
    scaler = torch.cuda.amp.GradScaler()
    val_acc_list=[]
    epoch_best = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scheduler, scaler, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        val_acc_list.append(acc1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model_version,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : _optimizer.state_dict(),
            }, is_best, args)
        if is_best:
            epoch_best = epoch

    checkpoint = torch.load(f'{args.identifier}_model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # evaluate on test set
    acc_test = validate(test_loader, model, criterion, args)

    with open(os.path.join(f"{args.identifier}.txt"), 'w') as f:
        f.write(f'test acc (%): {acc_test}\n')
        f.write(f'best val acc (%): {best_acc1}\n')
        f.write(f'best epoch: {epoch_best}\n\n')
        f.write(f'val history: {val_acc_list}\n')

    with open(os.path.join(f"settings_{args.identifier}.txt"), 'w') as f:
        f.write(f'args: {args}')



def train(train_loader, model, criterion, optimizer, epoch, scheduler, scaler, args):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1],
    #     prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        #  FP-16 사용
        with torch.cuda.amp.autocast():
            # compute output
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        # acc1, _ = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

        scheduler.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)


def validate(val_loader, model, criterion, args):
    # batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            with torch.cuda.amp.autocast():
                # compute output
                output = model(images)
                # loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            # losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg.item()


def save_checkpoint(state, is_best, args):
    # torch.save(state, f'{args.identifier}_checkpoint.pth.tar')
    if is_best:
        torch.save(state, f'{args.identifier}_model_best.pth.tar')
        # shutil.copyfile(f'{args.identifier}_checkpoint.pth.tar', f'{args.identifier}_model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
