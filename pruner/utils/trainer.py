import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import shutil
import copy
import math

from dataset import Data
import torch.distributed as dist
import numpy as np
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import distribute_bn
from timm.utils import reduce_tensor

pjoin = os.path.join


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


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt='%.6f'):
        self.name = name
        self.fmt = fmt
        self.max = self.min = self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

        self.avg = self.sum / self.count
        self.max = self.val if self.val > val else val
        self.min = val if self.val > val else self.val

    def __str__(self):
        # try:
        #     s = self.name + " : " + self.fmt % self.avg
        # except: # for sure.
        #     s = self.name + " : " + self.fmt.replace(':', '%') % self.avg
        s = self.name + " : " + self.fmt % self.avg
        return s


class Trainer(object):
    def __init__(self, args, logger, trian_epochs, lr, warmup_epochs = 0, cooldown_epochs = 0):
        super(Trainer, self).__init__()

        self.args = args
        self.epochs = trian_epochs
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs

        self.dataset = args.dataset
        self.loader = Data(args)
        self.train_loader = self.loader.train_loader
        self.val_loader = self.loader.test_loader
        ##self.criterion = nn.CrossEntropyLoss().cuda() 할지??
        self.criterion = nn.CrossEntropyLoss()

        self.logprint = logger.log_printer.logprint
        self.accprint = logger.log_printer.accprint
        self.netprint = logger.netprint
        self.ExpID = logger.ExpID
        self.weights_path = logger.weights_path

    def train_iterativeKD(self, model,print_log=True, teacher_model = None):
        optimizer = torch.optim.SGD(model.parameters(), self.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        lr_scheduler, num_epochs = create_scheduler(self.args, self.epochs, self.cooldown_epochs,
                                                    self.warmup_epochs, optimizer)

        epochs_per_iter = num_epochs / self.args.KD_iter_step if self.args.KD_epochs_per_iter is None else self.args.KD_epochs_per_iter

        acc1_list, loss_test_list = [], []
        best_acc1 = 0
        best_acc1_epoch = 0

        assert teacher_model is not None

        for iter in range( self.args.KD_iter_step):

            for epoch in range(epochs_per_iter):
                self.train_epoch_selfDistill(model, teacher_model,
                                             optimizer, epoch, lr_scheduler,
                                             alpha=self.args.KD_alpha, temperature=self.args.KD_temporature)

                acc1, acc5, loss_test = self.validate(model)
                acc1_list.append(acc1)
                loss_test_list.append(loss_test)
                lr_scheduler.step(epoch + 1)

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                if is_best:
                    best_acc1_epoch = epoch
                    best_loss_test = loss_test
                if print_log:
                    lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    self.accprint(
                        "Acc1 %.4f Acc5 %.4f Loss_test %.4f | Epoch %d (Best_Acc1 %.4f @ Best_Acc1_Epoch %d) lr %s" %
                        (acc1, acc5, loss_test, epoch, best_acc1, best_acc1_epoch, lr))

                # @mst: use our own save func
                state = {'epoch': epoch + 1,
                         'arch': self.args.backbone,
                         'state_dict': model.state_dict(),
                         'acc1': acc1,
                         'acc5': acc5,
                         'optimizer': optimizer.state_dict(),
                         'ExpID': self.ExpID,
                         'prune_state': 'finetune',
                         }

                self.save_model(state, is_best)

            teacher_model = copy.deepcopy(model)

            best = [best_acc1, best_loss_test]
            return best

    def train(self, model, print_log=True, teacher_model = None,  model_ema = None):
        optimizer = create_optimizer(self.args, model)
        lr_scheduler, num_epochs = create_scheduler(self.args, optimizer)

        acc1_list, loss_test_list = [], []
        self.best_acc1 = 0
        self.best_ema_acc1 = 0
        best_acc1_epoch = 0
        best_loss_test = 0

        for epoch in range(num_epochs):
            if self.args.distributed:
                self.loader.train_sampler.set_epoch(epoch)

            if teacher_model is None:
                self.train_epoch(self.loader.train_loader, model, self.criterion, optimizer, lr_scheduler, epoch, self.args, model_ema = model_ema)
            else:
                self.train_epoch_selfDistill(model, teacher_model,optimizer, epoch, lr_scheduler, alpha=self.args.KD_alpha, temperature=self.args.KD_temporature)

            if self.args.distributed:
                torch.cuda.synchronize()
                distribute_bn(model, self.args.world_size, True)
                if self.args.model_ema:
                    distribute_bn(model_ema, self.args.world_size, True)

            print('validate start')
            #def validate(self, loaders, model, criterion, epoch, args, nr_random_sample=0, alpha=1, batchnorm_calibration=True):
            acc1, acc5, loss_test = self.validate((self.loader.train_loader, self.loader.test_loader), model, self.criterion, epoch, self.args, batchnorm_calibration=True)

            if self.args.distributed:
                torch.cuda.synchronize()
                distribute_bn(model, self.args.world_size, True)
                if self.args.model_ema:
                    distribute_bn(model_ema, self.args.world_size, True)


            acc1_list.append(acc1)
            loss_test_list.append(loss_test)

            lr_scheduler.step(epoch+1, loss_test)

            # remember best acc@1 and save checkpoint
            if self.args.local_rank == 0 and self.args.rank == 0:
                is_best = acc1 > self.best_acc1
                self.best_acc1 = max(acc1, self.best_acc1)
                if is_best:
                    best_acc1_epoch = epoch
                    best_loss_test = loss_test
                if print_log and self.args.local_rank==0:
                    lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    self.accprint( "Acc1 %.4f Acc5 %.4f Loss_test %.4f | Epoch %d (Best_Acc1 %.4f @ Best_Acc1_Epoch %d) lr %s" %
                        (acc1, acc5, loss_test, epoch, self.best_acc1, best_acc1_epoch, lr))

                # @mst: use our own save func
                    state = {'epoch': epoch + 1,
                             'arch': self.args.backbone,
                             'state_dict': model.state_dict(),
                             'acc1': acc1,
                             'acc5': acc5,
                             'optimizer': optimizer.state_dict(),
                             'ExpID': self.ExpID,
                             'prune_state': 'finetune',
                             }
                    if self.args.model_ema:
                        state['state_dict_ema'] = model_ema.state_dict()

                    self.save_model(state, is_best)
                    
                    if int(acc1*10) == 762:
                        filename = pjoin(self.weights_path,
                              f"{self.args.backbone}_{self.args.dataset}_762.pth")
                        torch.save( state, filename )

        if self.args.model_ema:
            ema_acc1, ema_acc5, ema_loss_test = self.validate((self.loader.train_loader, self.loader.test_loader), model_ema.module,
                                                  self.criterion, 0, self.args, batchnorm_calibration=False)
            print("EMA Model Accuracy Acc1 %.4f Acc5 %.4f Loss_test %.4f" % (ema_acc1, ema_acc5, ema_loss_test))

        best = [self.best_acc1, best_loss_test]

        return best

    def train_epoch(self, train_loader, model, criterion, optimizer, lr_scheduler, epoch, args, model_ema = None):
        meter = {
            'loss': AverageMeter('loss'),
            'top1': AverageMeter('top1'),
            'top5': AverageMeter('top5'),
            'batch_time': AverageMeter('batch_time')
        }

        total_sample = len(train_loader.sampler)
        batch_size = args.batch_size
        progress = None
        if args.local_rank == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            progress = ProgressMeter(len(self.train_loader), [meter['loss'], meter['top1'], meter['top5'], f'learning rate : {lr:.8f}'],  prefix="Epoch: [{}]".format(epoch))

        if args.local_rank == 0:
            print(f'Training: {total_sample} samples ({batch_size} per mini-batch)')
        num_updates = epoch * len(train_loader)
        model.train()

        for batch_idx, (images, target) in enumerate(train_loader):
            # measure data loading time
            images = images.to(self.args.device)
            target = target.to(self.args.device)

            optimizer.zero_grad()
            # compute output
            start_time = time.time()
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = self.accuracy(output.detach(), target, topk=(1, 5))
            self.update_meter(meter, loss, acc1, acc5, images.size(0), time.time() - start_time, args.world_size)

            # compute gradient and do SGD step
            loss.backward()

            optimizer.step()
            num_updates += 1

            if model_ema is not None:
                model_ema.update(model)

            if lr_scheduler is not None:
                lr_scheduler.step_update( num_updates=num_updates, metric=meter['loss'].avg)

            if args.local_rank == 0 and (batch_idx + 1) % 10 == 0 and progress is not None:
                progress.display(batch_idx)

        return meter['top1'].avg, meter['top5'].avg, meter['loss'].avg


    def train_epoch_selfDistill(self, model, teacher_model, optimizer, epoch, lr_scheduler, alpha=0.5, temperature = 30.0, print_log=True):
        """
        [ICCAS 2023] Iterative Pruning of Neural Network with Self-Distillation (Author: Dr. Jonghee Park @ KETI)
        """
        batch_time = AverageMeter('Time', '%6.3f')
        data_time = AverageMeter('Data', '%6.3f')
        losses = AverageMeter('Loss', '%.4e')
        top1 = AverageMeter('Acc@1', '%6.2f')
        top5 = AverageMeter('Acc@5', '%6.2f')
        progress = ProgressMeter(len(self.train_loader), [batch_time, data_time, losses, top1, top5],
                                 prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        num_updates = epoch * len(self.train_loader)

        for i, (images, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = self.criterion(output, target)

            with torch.no_grad():
                teacher_model.eval()
                teacher_output = teacher_model(images)
                soft_targets = F.softmax(teacher_output / temperature, dim=1)

            distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output / temperature, dim=1), soft_targets)*(temperature * temperature * 2.0 + alpha) + loss * (1-alpha)

            # measure accuracy and record loss
            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            distillation_loss.backward()
            optimizer.step()

            num_updates += 1
            lr_scheduler.step_update(num_updates=num_updates)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if print_log and i % 10 == 0:
                progress.display(i)

    def update_meter(self, meter, loss, acc1, acc5, size, batch_time, world_size):
        if self.args.distributed:
            reduced_loss = reduce_tensor(loss.data, world_size)
            reduced_top1 = reduce_tensor(acc1, world_size)
            reduced_top5 = reduce_tensor(acc5, world_size)
            meter['loss'].update(reduced_loss.item(), size)
            meter['top1'].update(reduced_top1.item(), size)
            meter['top5'].update(reduced_top5.item(), size)
            meter['batch_time'].update(batch_time)
        else:
            meter['loss'].update(loss, size)
            meter['top1'].update(acc1, size)
            meter['top5'].update(acc5, size)
            meter['batch_time'].update(batch_time)

    def start_cal_bn(self, model, train_loader, args, cal_limit=100):
        print('bn calibration start')
        model.eval()
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.training = True
                m.momentum = None
        n = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if n > cal_limit:
                break

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            model(inputs)
            n += 1
        print('bn calibration end')

    def validate(self, loaders, model, criterion, epoch, args, nr_random_sample=0, alpha=1, batchnorm_calibration=True):
        train_loader, test_loader = loaders

        if batchnorm_calibration:
            calibration_samples = 6000
            self.start_cal_bn(model, train_loader=train_loader, args=args, cal_limit=calibration_samples // args.batch_size)



        meter = {
            'loss': AverageMeter('loss'),
            'top1': AverageMeter('top1'),
            'top5': AverageMeter('top5'),
            'batch_time': AverageMeter('batch_time')
        }

        total_sample = len(test_loader.sampler)
        batch_size = self.args.batch_size
        steps_per_epoch = math.ceil(total_sample / batch_size)
        progress = None

        if args.local_rank == 0:
            print(f'Validation: {total_sample} samples ({batch_size} per mini-batch)')
            progress = ProgressMeter(len(test_loader),
                                     [meter['loss'], meter['top1'], meter['top5']],
                                     prefix="Epoch: [{}]".format(epoch))

        model.eval()

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            with torch.no_grad():
                if not args.prefetcher:
                    inputs = inputs.to(args.device)
                    targets = targets.to(args.device)

                start_time = time.time()
                outputs = model(inputs)

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    outputs = outputs.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    targets = targets[0:targets.size(0):reduce_factor]

                loss = alpha * criterion(outputs, targets)

                acc1, acc5 = self.accuracy(outputs.data, targets.data, topk=(1, 5))
                self.update_meter(meter, loss, acc1, acc5, inputs.size(0), time.time() - start_time, args.world_size)

                if args.local_rank == 0 and (batch_idx+1) % 10 == 0 and progress is not None:
                    progress.display(batch_idx)

        if args.local_rank == 0:
            print('Test set accuracy: Top1 Acc: {} Top5 Acc : {} Loss : {}'.format(meter['top1'].avg, meter['top5'].avg, meter['loss'].avg))


        return meter['top1'].avg, meter['top5'].avg, meter['loss'].avg

    def save_checkpoint(self,state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    # @mst: use our own save model function
    def save_model(self, state, is_best=False, mark=''):
        if self.args.channel_rounding is not None:
            out = pjoin(self.weights_path, f"{self.args.backbone}_{self.args.dataset}_prune_ratio_{self.args.prune_ratio}_{self.args.channel_rounding}ch.pth")
        else:
            out = pjoin(self.weights_path,
                        f"{self.args.backbone}_{self.args.dataset}_prune_ratio_{self.args.prune_ratio}.pth")
        torch.save(state, out)
        if is_best:
            if self.args.channel_rounding is not None:
                out_best = pjoin(self.weights_path, f"{self.args.backbone}_{self.args.dataset}_prune_ratio_{self.args.prune_ratio}_{self.args.channel_rounding}ch_best.pth")
            else:
                out_best = pjoin(self.weights_path,
                                 f"{self.args.backbone}_{self.args.dataset}_prune_ratio_{self.args.prune_ratio}_best.pth")
            torch.save(state, out_best)
        if mark:
            out_mark = pjoin(self.weights_path, "checkpoint_{}.pth".format(mark))
            torch.save(state, out_mark)

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        maxk = min(max(topk), output.size()[1])
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
