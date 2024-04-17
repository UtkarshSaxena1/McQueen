import argparse
from cmath import pi
import datetime
import hashlib
import os
import random
import shutil
import time
import warnings
import copy
import math
import google.protobuf as pb
import google.protobuf.text_format
import models._modules as my_nn
import numpy as np
import plotly.graph_objects as go
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from proto import efficient_pytorch_pb2 as eppb
from pytorchcv.model_provider import get_model as ptcv_get_model
from tensorboardX import SummaryWriter
from utils import wrapper
from utils.ptflops import get_model_complexity_info
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from typing import Union

str_q_mode_map = {eppb.Qmode.layer_wise: my_nn.Qmodes.layer_wise,
                  eppb.Qmode.kernel_wise: my_nn.Qmodes.kernel_wise}


def get_base_parser():
    """
        Default values should keep stable.
    """

    print('Please do not import ipdb when using distributed training')

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--hp', type=str,
                        help='File path to save hyperparameter configuration')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume-after', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--bn-fusion', action='store_true', default=False,
                        help='ConvQ + BN fusion')
    parser.add_argument('--resave', action='store_true', default=False,
                        help='resave the model')

    parser.add_argument('--gen-layer-info', action='store_true', default=False,
                        help='whether to generate layer information for latency evaluation on hardware')

    parser.add_argument('--print-histogram', action='store_true', default=False,
                        help='save histogram of weight in tensorboard')
    parser.add_argument('--freeze-bn', action='store_true',
                        default=False, help='Freeze BN')
    return parser


def main_s1_set_seed(hp):
    if hp.HasField('seed'):
        random.seed(hp.seed)
        torch.manual_seed(hp.seed)
        cudnn.deterministic = True
        np.random.seed(hp.seed)
        torch.cuda.manual_seed(hp.seed)
        cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(hp.seed)
        torch.cuda.manual_seed_all(hp.seed)

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')



def sync_tensor_across_gpus(t: Union[torch.Tensor, None]
                            ) -> Union[torch.Tensor, None]:
    # t needs to have dim 0 for troch.cat below. 
    # if not, you need to prepare it.
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t)  # this works with nccl backend when tensors need to be on gpu. 
   # for gloo and mpi backends, tensors need to be on cpu. also this works single machine with 
   # multiple   gpus. for multiple nodes, you should use dist.all_gather_multigpu. both have the 
   # same definition... see [here](https://pytorch.org/docs/stable/distributed.html). 
   #  somewhere in the same page, it was mentioned that dist.all_gather_multigpu is more for 
   # multi-nodes. still dont see the benefit of all_gather_multigpu. the provided working case in 
   # the doc is  vague... 
    return gather_t_tensor

def main_s2_start_worker(main_worker, args, hp):
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    args.world_size = hp.multi_gpu.world_size
    if hp.HasField('multi_gpu') and hp.multi_gpu.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or (hp.HasField(
        'multi_gpu') and hp.multi_gpu.multiprocessing_distributed)

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: {}'.format(ngpus_per_node))
    if hp.HasField('multi_gpu') and hp.multi_gpu.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def get_hyperparam(args):
    assert os.path.exists(args.hp)
    hp = eppb.HyperParam()
    with open(args.hp, 'r') as rf:
        pb.text_format.Merge(rf.read(), hp)
    return hp

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2])
                        for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')
    # TODO; if no gpu, return None
    try:
        visible_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        memory_visible = []
        for i in visible_gpu.split(','):
            memory_visible.append(memory_available[int(i)])
        return np.argmax(memory_visible)
    except KeyError:
        return np.argmax(memory_available)



def get_lr_scheduler(optimizer, lr_domain):
    """
    Args:
        optimizer:
        lr_domain ([proto]): [lr configuration domain] e.g. args.hp args.hp.bit_pruner
    """
    if isinstance(lr_domain, argparse.Namespace):
        lr_domain = lr_domain.hp
    if lr_domain.lr_scheduler == eppb.LRScheduleType.CosineAnnealingLR:
        print('Use cosine scheduler')
        scheduler_next = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=lr_domain.epochs)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.StepLR:
        print('Use step scheduler, step size: {}, gamma: {}'.format(
            lr_domain.step_lr.step_size, lr_domain.step_lr.gamma))
        scheduler_next = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_domain.step_lr.step_size, gamma=lr_domain.step_lr.gamma)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.MultiStepLR:
        print('Use MultiStepLR scheduler, milestones: {}, gamma: {}'.format(
            lr_domain.multi_step_lr.milestones, lr_domain.multi_step_lr.gamma))
        scheduler_next = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_domain.multi_step_lr.milestones, gamma=lr_domain.multi_step_lr.gamma)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.CyclicLR:
        print('Use CyclicLR scheduler')
        if not lr_domain.cyclic_lr.HasField('step_size_down'):
            step_size_down = None
        else:
            step_size_down = lr_domain.cyclic_lr.step_size_down

        cyclic_mode_map = {eppb.CyclicLRParam.Mode.triangular: 'triangular',
                           eppb.CyclicLRParam.Mode.triangular2: 'triangular2',
                           eppb.CyclicLRParam.Mode.exp_range: 'exp_range', }

        scheduler_next = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=lr_domain.cyclic_lr.base_lr, max_lr=lr_domain.cyclic_lr.max_lr,
            step_size_up=lr_domain.cyclic_lr.step_size_up, step_size_down=step_size_down,
            mode=cyclic_mode_map[lr_domain.cyclic_lr.mode], gamma=lr_domain.cyclic_lr.gamma)
    else:
        raise NotImplementedError
    
    return scheduler_next

def get_optimizer(model, args):
    # define optimizer after process model
    print('define optimizer')
    if args.hp.optimizer == eppb.OptimizerType.SGD:
        params = add_weight_decay(model, weight_decay=args.hp.sgd.weight_decay,
                                  skip_keys=['expand_', 'running_scale', 'alpha_a', 'alpha_w',
                                             'standard_threshold', 'bits_w', 'bits_a', 'pruning_threshold', 'beta_w', 'beta_a'])
        optimizer = torch.optim.SGD(params, args.hp.lr,
                                    momentum=args.hp.sgd.momentum)
        print('Use SGD')
    elif args.hp.optimizer == eppb.OptimizerType.Adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.hp.lr, weight_decay=args.hp.adam.weight_decay)
        print('Use Adam')
    else:
        raise NotImplementedError
    return optimizer


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

def accuracy_ee(output, confidence, target, topk=(1,)):
    """Computes the accuracy of the max prediction above confidence threshold"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        output_confidence = torch.nn.functional.softmax(output)

        conf, pred = output_confidence.topk(maxk, 1, True, True)
        pred = pred.t()
        conf = conf.t()
        correct = torch.logical_and((pred.eq(target.view(1, -1).expand_as(pred))), conf.ge(confidence))
        # correct = torch.logical_and(torch.logical_not(pred.eq(target.view(1, -1).expand_as(pred))), conf.ge(confidence))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_ee_overall(output, output_exit, confidence, target):
    """Computes the accuracy of the max prediction above confidence threshold"""
    with torch.no_grad():
        batch_size = target.size(0)
        
        #find accuracy of examples which exit early
        exit_confidence = torch.nn.functional.softmax(output_exit, dim=1)
        conf, pred = exit_confidence.topk(1, 1, True, True)
        pred = pred.t()
        conf = conf.t()
        correct_exit = torch.logical_and((pred.eq(target.view(1, -1).expand_as(pred))), conf.ge(confidence))
        total_correct = correct_exit.reshape(-1).float().sum(0, keepdim=True)
        #incorrect exit when high confidence but wrong answer
        incorrect_exit = torch.logical_and(torch.logical_not(pred.eq(target.view(1, -1).expand_as(pred))), conf.ge(confidence))
        total_incorrect = incorrect_exit.reshape(-1).float().sum(0,keepdim=True).mul_(100.0/batch_size)
        #find accuracy of examples which travel to the end
        mask = exit_confidence.max(1)[0].le(confidence).float()
        
        output_end = output[mask==1]
        target_end = target[mask==1]
        _, pred_end = output_end.topk(1, 1, True, True)
        pred_end = pred_end.t()
        correct_end = pred_end.eq(target_end.view(1, -1).expand_as(pred_end))
        total_correct += correct_end.reshape(-1).float().sum(0, keepdim=True)

        res_overall = total_correct.mul_(100.0/batch_size)

        num_exit = exit_confidence.max(1)[0].ge(confidence).float().sum()

        return res_overall, num_exit, total_incorrect

def accuracy_ee_ensembles(output, idx, target):
    """Computes the accuracy of the max prediction above confidence threshold"""
    with torch.no_grad():
        batch_size = target.size(0)
        coeffs = torch.tensor([0.54, 0.64, 0.77, 0.87, 1])
        
        #find accuracy at a particular exit 
        #find sum of probabilities from preceeding heads and choose the sample with max probability
        
        for i in range(idx+1):
            if i ==0 :
                exit_confidence = coeffs[0] * torch.nn.functional.softmax(output[0], dim=1)
            else: 
                exit_confidence += coeffs[idx] * torch.nn.functional.softmax(output[i], dim=1)

        conf, pred = exit_confidence.topk(1, 1, True, True)
        pred = pred.t()
        conf = conf.t()
        correct_exit = (pred.eq(target.view(1, -1).expand_as(pred)))
        total_correct = correct_exit.reshape(-1).float().sum(0, keepdim=True)

        res_overall = total_correct.mul_(100.0/batch_size)


        return res_overall

def accuracy_ee_overall_patience(output, output_exit, confidence, patience_th, target):
    """Computes the accuracy of the max prediction above confidence threshold"""
    with torch.no_grad():
        batch_size = target.size(0)
        exits = torch.zeros(len(output_exit))
        correct_exit = 0
        incorrect_exit = 0
        #find accuracy of examples which exit early
        for j in range(output.shape[0]):
            patience = torch.zeros(10)
            early_exit = 0
            target_cur = target[j]

            for i in range(len(output_exit)-1):
                exit_confidence = torch.nn.functional.softmax(output_exit[i][j].unsqueeze(0), dim=1)
                max_conf, max_pred = exit_confidence.max(1)
                if max_conf > confidence:
                    patience[max_pred] +=1
                
                
                if((patience.ge(patience_th)).float().sum() >0):
                    #exit

                    exits[i] +=1
                    _,pred_cur = patience.unsqueeze(0).max(1)
                    pred_cur = pred_cur.to(target_cur.device)
                    
                    if(pred_cur == target_cur):
                        correct_exit +=1
                    else:
                        incorrect_exit +=1
                    early_exit = 1
                    break
            if(early_exit !=1):
                _,pred_cur = output[j].unsqueeze(0).max(1)
                pred_cur = pred_cur.to(target_cur.device)
                exits[len(output_exit)-1] +=1
                if(pred_cur == target_cur):
                    correct_exit +=1
                else:
                    incorrect_exit +=1



        res_overall = torch.tensor(correct_exit * (100.0/batch_size))
        total_incorrect = torch.tensor(incorrect_exit * (100.0/batch_size))

        return res_overall, exits, total_incorrect

def accuracy_ee_overall_patience_v2(output, output_exit, confidence2, patience_th, target):
    """Computes the accuracy of the max prediction above confidence threshold"""
    with torch.no_grad():
        batch_size = target.size(0)
        exits = torch.zeros(len(output_exit))
        correct_exit = 0
        incorrect_exit = torch.zeros(len(output_exit))
        #find accuracy of examples which exit early
        for j in range(output.shape[0]):
            patience = torch.zeros(1000)
            early_exit = 0
            target_cur = target[j]

            for i in range(len(output_exit)-1):
                exit_confidence = torch.nn.functional.softmax(output_exit[i][j].unsqueeze(0), dim=1)
                max_conf, max_pred = exit_confidence.max(1)
                
                if max_conf > confidence2:
                    patience[max_pred] +=1
                if((patience.ge(patience_th)).float().sum() >0):
                    #exit with patience

                    exits[i] +=1
                    _,pred_cur = patience.unsqueeze(0).max(1)
                    pred_cur = pred_cur.to(target_cur.device)
                    
                    if(pred_cur == target_cur):
                        correct_exit +=1
                    else:
                        incorrect_exit[i] +=1
                    early_exit = 1
                    break
            if(early_exit !=1):
                _,pred_cur = output[j].unsqueeze(0).max(1)
                pred_cur = pred_cur.to(target_cur.device)
                exits[len(output_exit)-1] +=1
                if(pred_cur == target_cur):
                    correct_exit +=1
                else:
                    incorrect_exit[-1] +=1



        res_overall = torch.tensor(correct_exit * (100.0/batch_size))
        total_incorrect = incorrect_exit.sum() * (100.0/batch_size)

        return res_overall, exits, total_incorrect, incorrect_exit

def add_weight_decay(model, weight_decay, skip_keys):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                no_decay.append(param)
                added = True
                break
        if not added:
            # if ('exit1.linear.weight' in name):
            #     continue
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def set_bn_eval(m):
    """[summary]
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
    https://github.com/pytorch/pytorch/issues/16149
        requires_grad does not change the train/eval mode, 
        but will avoid calculating the gradients for the affine parameters (weight and bias).
        bn.train() and bn.eval() will change the usage of the running stats (running_mean and running_var).
    For detailed computation of Batch Normalization, please refer to the source code here.
    https://github.com/pytorch/pytorch/blob/83c054de481d4f65a8a73a903edd6beaac18e8bc/torch/csrc/jit/passes/graph_fuser.cpp#L232
    The input is normalized by the calculated mean and variance first. 
    Then the transformation of w*x+b is applied on it by adding the operations to the computational graph.
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
    return


def set_bn_grad_false(m):
    """freeze \gamma and \beta in BatchNorm
        model.apply(set_bn_grad_false)
        optimizer = SGD(model.parameters())
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if m.affine:
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)


def set_param_grad_false(model):
    for name, param in model.named_parameters():  # same to set bn val? No
        if param.requires_grad:
            param.requires_grad_(False)
            print('frozen weights. shape:{}'.format(param.shape))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu,non_blocking=True)
                target = target.cuda(args.gpu,non_blocking=True)
            # compute output
            output = model(input)
            if isinstance(output, list):
                output = output[-1]
            loss = criterion(output, target).mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0:
                progress.print(i)
            if args.hp.overfit_test:
                break

        print(' *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(time=batch_time, top1=top1, top5=top5))

    return top1.avg, top5.avg


def validate_ee(val_loader, model, criterion, args, confidence=0.9, idx = -1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_overall = AverageMeter('Acc_e@1', ':6.2f')
    top1_exit = AverageMeter('Acc_e@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_misclassified = AverageMeter('Acc_e@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #samples exited
    samples_exit = 0
    samples_total = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu,non_blocking=True)
                target = target.cuda(args.gpu,non_blocking=True)
            # compute output
            # output_exit, output = model(input)
            output_exit = model(input)
            output = output_exit[idx]
            loss = (criterion(output, target)).mean()
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_exit, _ = accuracy(output_exit[idx], target, topk=(1,5))

            acc1_overall, num_exit, exit_missclassified = accuracy_ee_overall(output, output_exit[idx], confidence, target)
            samples_exit += num_exit
            samples_total += output.size()[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            top1_overall.update(acc1_overall[0], input.size(0))
            top1_exit.update(acc1_exit[0], input.size(0))
            top1_misclassified.update(exit_missclassified[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0:
                progress.print(i)
            if args.hp.overfit_test:
                break
            
        percentage_samples_exited = 100.0 * samples_exit / samples_total
        print(' *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} || Acc_exit@1 {top1_exit.avg:.3f} Acc_overall@1 {top1_overall.avg:.3f} Exit percentage {percentage_samples_exited:.3f} Exit Missclassified {top1_misclassified.avg:.3f}'
              .format(time=batch_time, top1=top1, top5=top5, top1_exit=top1_exit, top1_overall=top1_overall, percentage_samples_exited=percentage_samples_exited, top1_misclassified=top1_misclassified))
    return top1.avg, top5.avg, top1_overall.avg, percentage_samples_exited


def validate_ee_ensembles(val_loader, model, ensemble_model, criterion, args, idx = -1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_overall = AverageMeter('Acc_e@1', ':6.2f')
    top1_exit = AverageMeter('Acc_e@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_misclassified = AverageMeter('Acc_e@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #samples exited
    samples_exit = 0
    samples_total = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu,non_blocking=True)
                target = target.cuda(args.gpu,non_blocking=True)
            # compute output
            # output_exit, output = model(input)
            # output_exit = model(input)
            # output_exit = torch.stack(output_exit, dim = 1)
            # output_exit = output_exit[:,:(idx+1),:]
            output_ensemble = ensemble_model(input)
            loss = criterion(output_ensemble.log(), target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output_ensemble, target, topk=(1, 5))

            # acc1_ensemble = accuracy_ee_ensembles(output_exit, idx, target)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            # top1_overall.update(acc1_ensemble[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0:
                progress.print(i)
            if args.hp.overfit_test:
                break
            
        print(' *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} || Acc_exit@1 {top1_exit.avg:.3f} Acc_ensemble@1 {top1_overall.avg:.3f}'
              .format(time=batch_time, top1=top1, top5=top5, top1_exit=top1_exit, top1_overall=top1_overall))
    return top1.avg, top5.avg, top1_overall.avg


def validate_ee_patience(val_loader, model, criterion, args, confidence, patience,idx = -1, num_exits = 1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_overall = AverageMeter('Acc_e@1', ':6.2f')
    top1_exit = AverageMeter('Acc_e@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_misclassified = AverageMeter('Acc_e@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #samples exited
    samples_exit = 0
    samples_total = 0
    exits = torch.zeros(num_exits)
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu,non_blocking=True)
                target = target.cuda(args.gpu,non_blocking=True)
            # compute output
            # output_exit, output = model(input)
            output_exit = model(input)
            output = output_exit[-1]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_exit, _ = accuracy(output_exit[idx], target, topk=(1,5))
            # acc1_exit, _ = accuracy(output_exit[idx], target, topk=(1,5))
            acc1_overall, num_exit, exit_missclassified = accuracy_ee_overall_patience(output, output_exit, confidence,patience, target)
            # acc1_overall, num_exit, exit_missclassified = accuracy_ee_overall(output, output_exit, confidence, target)
            samples_exit += num_exit[0:-1].sum()
            samples_total += output.size()[0]
            exits += num_exit
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            top1_overall.update(acc1_overall, input.size(0))
            top1_exit.update(acc1_exit[0], input.size(0))
            top1_misclassified.update(exit_missclassified, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0:
                progress.print(i)
            if args.hp.overfit_test:
                break
        percentage_samples_exited = 100.0 * samples_exit / samples_total
        print(' *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} || Acc_exit@1 {top1_exit.avg:.3f} Acc_overall@1 {top1_overall.avg:.3f} Exit percentage {percentage_samples_exited:.3f} Exit Missclassified {top1_misclassified.avg:.3f}'
              .format(time=batch_time, top1=top1, top5=top5, top1_exit=top1_exit, top1_overall=top1_overall, percentage_samples_exited=percentage_samples_exited, top1_misclassified=top1_misclassified))
    return top1.avg, top5.avg, top1_overall.avg, percentage_samples_exited, exits*100/samples_total

def validate_ee_patience_v2(val_loader, model, ensemble_model, criterion, args, confidence, patience,idx = -1, num_exits = 1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_overall = AverageMeter('Acc_e@1', ':6.2f')
    top1_exit = AverageMeter('Acc_e@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_misclassified = AverageMeter('Acc_e@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #samples exited
    samples_exit = 0
    samples_total = 0
    samples_incorrect = torch.zeros(num_exits)
    exits = torch.zeros(num_exits)
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu,non_blocking=True)
                target = target.cuda(args.gpu,non_blocking=True)
            
            # compute output
            output_exit = model(input)
            if ensemble_model is not None:
                output = ensemble_model[-1](input)
            else:
                output = output_exit[-1]
            loss = (criterion(output, target)).mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_exit, _ = accuracy(output_exit[idx], target, topk=(1,5))
            # acc1_exit, _ = accuracy(output_exit[idx], target, topk=(1,5))
            acc1_overall, num_exit, exit_missclassified, incorrect = accuracy_ee_overall_patience_v2(output, output_exit, confidence,patience, target)
            samples_exit += num_exit[0:-1].sum()
            samples_total += output.size()[0]
            exits += num_exit
            samples_incorrect += incorrect
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            top1_overall.update(acc1_overall, input.size(0))
            top1_exit.update(acc1_exit[0], input.size(0))
            top1_misclassified.update(exit_missclassified, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0:
                progress.print(i)
            if args.hp.overfit_test:
                break
        percentage_samples_exited = 100.0 * samples_exit / samples_total
        print(' *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} || Acc_exit@1 {top1_exit.avg:.3f} Acc_overall@1 {top1_overall.avg:.3f} Exit percentage {percentage_samples_exited:.3f} Exit Missclassified {top1_misclassified.avg:.3f}'
              .format(time=batch_time, top1=top1, top5=top5, top1_exit=top1_exit, top1_overall=top1_overall, percentage_samples_exited=percentage_samples_exited, top1_misclassified=top1_misclassified))
    return top1.avg, top5.avg, top1_overall.avg, percentage_samples_exited, exits*100/samples_total, samples_incorrect*100/samples_total


def validate_2ee(val_loader, model, criterion, args, confidence, idx = 0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_overall = AverageMeter('Acc_e@1', ':6.2f')
    top1_exit1 = AverageMeter('Acc_e@1', ':6.2f')
    top1_exit2 = AverageMeter('Acc_e@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_misclassified = AverageMeter('Acc_e@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #samples exited
    samples_exit1 = 0
    samples_exit2 = 0
    samples_total = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu,non_blocking=True)
                target = target.cuda(args.gpu,non_blocking=True)
            # compute output
            output_exit, output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_exit1, _ = accuracy(output_exit[0], target, topk=(1,5))
            acc1_exit2, _ = accuracy(output_exit[1], target, topk=(1,5))
            # acc1_exit, _ = accuracy(output_exit[idx], target, topk=(1,5))

            # acc1_overall, num_exit, exit_missclassified = accuracy_ee_overall(output, output_exit[idx], confidence, target)
            acc1_overall, num_exit1, num_exit2, exit_missclassified = accuracy_2ee_overall(output, output_exit[0], output_exit[1], confidence, target)
            samples_exit1 += num_exit1 
            samples_exit2 += num_exit2
            samples_total += output.size()[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            top1_overall.update(acc1_overall[0], input.size(0))
            top1_exit1.update(acc1_exit1[0], input.size(0))
            top1_exit2.update(acc1_exit2[0], input.size(0))
            top1_misclassified.update(exit_missclassified[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0:
                progress.print(i)
            if args.hp.overfit_test:
                break
        percentage_samples_exited = 100.0 * (samples_exit1+samples_exit2) / samples_total
        percent_exit1 = 100.0 * samples_exit1 / samples_total
        percent_exit2 = 100.0 * samples_exit2 / samples_total
        print(' *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} || Acc_exit1@1 {top1_exit1.avg:.3f} || Acc_exit2@1 {top1_exit2.avg:.3f} Acc_overall@1 {top1_overall.avg:.3f} Exit percentage {percentage_samples_exited:.3f} Exit Missclassified {top1_misclassified.avg:.3f}'
              .format(time=batch_time, top1=top1, top5=top5, top1_exit1=top1_exit1, top1_exit2=top1_exit2, top1_overall=top1_overall, percentage_samples_exited=percentage_samples_exited, top1_misclassified=top1_misclassified))
    return top1.avg, top5.avg, top1_overall.avg, percent_exit1, percent_exit2

def validate_1x2ee(val_loader, model, criterion, args, confidence, idx = 0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_overall = AverageMeter('Acc_e@1', ':6.2f')
    top1_exit1 = AverageMeter('Acc_e@1', ':6.2f')
    top1_exit2 = AverageMeter('Acc_e@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_misclassified = AverageMeter('Acc_e@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #samples exited
    samples_exit1 = 0
    samples_exit2 = 0
    samples_total = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu,non_blocking=True)
                target = target.cuda(args.gpu,non_blocking=True)
            # compute output
            output_exit, output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_exit1, _ = accuracy(output_exit[0], target, topk=(1,5))
            acc1_exit2, _ = accuracy(output_exit[1], target, topk=(1,5))
            # acc1_exit, _ = accuracy(output_exit[idx], target, topk=(1,5))

            # acc1_overall, num_exit, exit_missclassified = accuracy_ee_overall(output, output_exit[idx], confidence, target)
            acc1_overall, num_exit1, exit_missclassified = accuracy_1x2ee_overall(output, output_exit[0], output_exit[1], confidence, target)
            samples_exit1 += num_exit1 
            samples_exit2 += torch.tensor(0)
            samples_total += output.size()[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            top1_overall.update(acc1_overall[0], input.size(0))
            top1_exit1.update(acc1_exit1[0], input.size(0))
            top1_exit2.update(acc1_exit2[0], input.size(0))
            top1_misclassified.update(exit_missclassified[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0:
                progress.print(i)
            if args.hp.overfit_test:
                break
        percentage_samples_exited = 100.0 * (samples_exit1+samples_exit2) / samples_total
        percent_exit1 = 100.0 * samples_exit1 / samples_total
        percent_exit2 = 100.0 * samples_exit2 / samples_total
        print(' *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} || Acc_exit1@1 {top1_exit1.avg:.3f} || Acc_exit2@1 {top1_exit2.avg:.3f} Acc_overall@1 {top1_overall.avg:.3f} Exit percentage {percentage_samples_exited:.3f} Exit Missclassified {top1_misclassified.avg:.3f}'
              .format(time=batch_time, top1=top1, top5=top5, top1_exit1=top1_exit1, top1_exit2=top1_exit2, top1_overall=top1_overall, percentage_samples_exited=percentage_samples_exited, top1_misclassified=top1_misclassified))
    return top1.avg, top5.avg, top1_overall.avg, percent_exit1, percent_exit2

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()

    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        #optimizer zero grad
        optimizer.zero_grad()
        
        # compute output
        output = model(inputs)
        loss_ce = criterion(output, targets)
        
        
        if (i == 0):
            total_elements_w = 0
            total_elements_a = 0
            for name, m in model.named_modules():
                if (name == 'conv1.0'):
                    #first layer activations to remain 8-bits and not trainable
                    total_elements_a = total_elements_a + m.num_elements
                    m.bits.data.copy_(torch.ones(1) * 8)
                    m.bits.requires_grad = False
                    continue
                elif('conv1.1' in name):
                    total_elements_w = total_elements_w + m.num_elements
                    
                elif('conv2.1' in name):
                    total_elements_w = total_elements_w + m.num_elements
                elif('shortcut.0.1' in name):
                    total_elements_w = total_elements_w + m.num_elements
                elif('linear.1' in name):
                    total_elements_w = total_elements_w + m.num_elements
                elif('conv1.0' in name):
                    total_elements_a = total_elements_a + m.num_elements
                elif('conv2.0' in name):
                    total_elements_a = total_elements_a + m.num_elements
                elif('linear.0' in name):
                    total_elements_a = total_elements_a + m.num_elements
                elif('shortcut.0.0' in name):
                    total_elements_a = total_elements_a + m.num_elements

        loss_reg_w = 0
        loss_reg_a = 0
        bits_per_weight = 0
        bits_per_activation = 0
        for name, m in model.named_modules():
            if('conv1.1' in name):
                loss_reg_w = loss_reg_w + (m.bits)**2 * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +  torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_w
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('conv2.1' in name):
                loss_reg_w = loss_reg_w + (m.bits)**2 * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_w
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('shortcut.0.1' in name):
                loss_reg_w = loss_reg_w + (m.bits)**2 * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_w
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('linear.1' in name):
                loss_reg_w = loss_reg_w + (m.bits)**2 * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_w
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif(name == 'conv1.0'):
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_a
                continue
            elif('conv1.0' in name):
                loss_reg_a = loss_reg_a + (m.bits)**2 * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_a 
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('conv2.0' in name):
                loss_reg_a = loss_reg_a + (m.bits)**2 * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements  / total_elements_a
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('linear.0' in name):
                loss_reg_a = loss_reg_a + (m.bits)**2 * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements  / total_elements_a
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('shortcut.0.0' in name):
                loss_reg_a = loss_reg_a + (m.bits)**2 * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements  / total_elements_a
                if(m.bits < 1):
                    m.bits.requires_grad = False
        loss_reg_w = args.hp.bit_penalty_w*loss_reg_w
        loss_reg_a = args.hp.bit_penalty_a*loss_reg_a
        loss = loss_ce + loss_reg_w + loss_reg_a
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        loss.backward()
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss, base_step + i)
            writer.add_scalar('loss/ce', loss_ce, base_step + i)
            writer.add_scalar('loss/reg_w', loss_reg_w, base_step + i)
            writer.add_scalar('loss/reg_a', loss_reg_a, base_step + i)
            writer.add_scalar('bits/bits_per_weight', bits_per_weight, base_step + i)
            writer.add_scalar('bits/bits_per_activation', bits_per_activation, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if('bits' in name):
            writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
        else:
            writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
        if param.grad is not None:
            writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    return bits_per_weight, bits_per_activation

def train_L1_penalty(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer, target_a, target_w):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    return_layers = {'conv1' : 'conv1',
                    'layer1.0.conv1' : 'layer1.0.conv1',
                    'layer1.0.conv2' : 'layer1.0.conv2',
                    'layer1.1.conv1' : 'layer1.1.conv1',
                    'layer1.1.conv2' : 'layer1.1.conv2',
                    'layer1.2.conv1' : 'layer1.2.conv1',
                    'layer1.2.conv2' : 'layer1.2.conv2',
                    'layer2.0.conv1' : 'layer2.0.conv1',
                    'layer2.0.conv2' : 'layer2.0.conv2',
                    'layer2.1.conv1' : 'layer2.1.conv1',
                    'layer2.1.conv2' : 'layer2.1.conv2',
                    'layer2.2.conv1' : 'layer2.2.conv1',
                    'layer2.2.conv2' : 'layer2.2.conv2',
                    'layer3.0.conv1' : 'layer3.0.conv1',
                    'layer3.0.conv2' : 'layer3.0.conv2',
                    'layer3.1.conv1' : 'layer3.1.conv1',
                    'layer3.1.conv2' : 'layer3.1.conv2',
                    'layer3.2.conv1' : 'layer3.2.conv1',
                    'layer3.2.conv2' : 'layer3.2.conv2',
                    'linear' : 'linear'}
    intermidiate_getter_student = MidGetter(model, return_layers=return_layers, keep_output=True)
    intermediate_getter_teacher = MidGetter(teacher_model, return_layers=return_layers, keep_output=True)
    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        #optimizer zero grad
        optimizer.zero_grad()
        
        # compute output
        # output = model(inputs)
        intermediate_outputs_student, output = intermidiate_getter_student(inputs)
        with torch.no_grad():
            intermediate_outputs_teacher, output_teacher = intermediate_getter_teacher(inputs)
        loss_ce = criterion(output, targets)
        
        
        if (i == 0):
            total_elements_w = 0
            total_elements_a = 0
            for name, m in model.named_modules():
                if (name == 'conv1.0'):
                    #first layer activations to remain 8-bits and not trainable
                    total_elements_a = total_elements_a + m.num_elements
                    m.bits.data.copy_(torch.ones(1) * 8)
                    m.bits.requires_grad = False
                    continue
                elif('conv1.1' in name):
                    total_elements_w = total_elements_w + m.num_elements
                    
                elif('conv2.1' in name):
                    total_elements_w = total_elements_w + m.num_elements
                elif('shortcut.0.1' in name):
                    total_elements_w = total_elements_w + m.num_elements
                elif('linear.1' in name):
                    total_elements_w = total_elements_w + m.num_elements
                elif('conv1.0' in name):
                    total_elements_a = total_elements_a + m.num_elements
                elif('conv2.0' in name):
                    total_elements_a = total_elements_a + m.num_elements
                elif('linear.0' in name):
                    total_elements_a = total_elements_a + m.num_elements
                elif('shortcut.0.0' in name):
                    total_elements_a = total_elements_a + m.num_elements

        penalty_w = 0
        penalty_a = 0
        bits_per_weight = 0
        bits_per_activation = 0
        for name, m in model.named_modules():
            if('conv1.1' in name):
                penalty_w = penalty_w + (m.bits) * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +  torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_w
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('conv2.1' in name):
                penalty_w = penalty_w + (m.bits) * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_w
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('shortcut.0.1' in name):
                penalty_w = penalty_w + (m.bits) * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_w
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('linear.1' in name):
                penalty_w = penalty_w + (m.bits) * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_w
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif(name == 'conv1.0'):
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_a
                continue
            elif('conv1.0' in name):
                penalty_a = penalty_a + (m.bits) * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements / total_elements_a 
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('conv2.0' in name):
                penalty_a = penalty_a + (m.bits) * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements  / total_elements_a
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('linear.0' in name):
                penalty_a = penalty_a + (m.bits) * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements  / total_elements_a
                if(m.bits < 1):
                    m.bits.requires_grad = False
            elif('shortcut.0.0' in name):
                penalty_a = penalty_a + (m.bits) * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements  / total_elements_a
                if(m.bits < 1):
                    m.bits.requires_grad = False
        loss_reg_w = args.hp.bit_penalty_w * (torch.abs(penalty_w - target_w))
        loss_reg_a = args.hp.bit_penalty_a * (torch.abs(penalty_a - target_a))
        loss_kd_inter = 0
        alpha_kd = 0.01
        T = 10
        alpha_kd_sce = 0.01
        for name, _ in intermediate_outputs_teacher.items():
            # print(name+':')
            # print("Student:" + str(torch.mean(intermediate_outputs_student[name]).item()))
            # print("Teacher:" + str(torch.mean(intermediate_outputs_teacher[name]).item()))
            teacher_tensor = intermediate_outputs_teacher[name] / torch.linalg.matrix_norm(intermediate_outputs_teacher[name], keepdim = True)
            student_tensor = intermediate_outputs_student[name] / torch.linalg.matrix_norm(intermediate_outputs_student[name], keepdim = True)
            loss_kd_inter =+ alpha_kd * torch.nn.functional.mse_loss(student_tensor, teacher_tensor)
        loss_kd_sce = alpha_kd_sce * torch.nn.functional.kl_div(torch.nn.functional.log_softmax(output/T, dim=1),
                             torch.nn.functional.softmax(output_teacher/T, dim=1)) * T * T
        loss = loss_ce + loss_reg_w + loss_reg_a + loss_kd_inter + loss_kd_sce
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        # compute gradient and do SGD step
        loss.backward()
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss, base_step + i)
            writer.add_scalar('loss/ce', loss_ce, base_step + i)
            writer.add_scalar('loss/reg_w', loss_reg_w, base_step + i)
            writer.add_scalar('loss/reg_a', loss_reg_a, base_step + i)
            writer.add_scalar('bits/bits_per_weight', bits_per_weight, base_step + i)
            writer.add_scalar('bits/bits_per_activation', bits_per_activation, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if('bits' in name):
            writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
        else:
            writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
        if param.grad is not None:
            writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    return bits_per_weight, bits_per_activation


def train_resnet20_kernelwise(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    
    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        #optimizer zero grad
        optimizer.zero_grad()
        
        for name, m in model.named_modules():
            if('conv1.1' in name):
                with torch.no_grad():
                    temp = m.bits
                    temp[temp < 1] = 1
                    m.bits.data.copy_(temp)
            elif('conv2.1' in name):
                with torch.no_grad():
                    temp = m.bits
                    temp[temp < 1] = 1
                    m.bits.data.copy_(temp)
            elif('shortcut.0.1' in name):
                with torch.no_grad():
                    temp = m.bits
                    temp[temp < 1] = 1
                    m.bits.data.copy_(temp)
        
        # compute output
        output = model(inputs)
        loss_ce = criterion(output, targets)
        
        
        if (i == 0):
            total_elements_w = 0
            total_elements_a = 0
            for name, m in model.named_modules():
                if (name == 'conv1.0'):
                    #first layer activations to remain 8-bits and not trainable
                    total_elements_a = total_elements_a + m.num_elements
                    m.bits.data.copy_(torch.ones(1) * 8)
                    m.bits.requires_grad = False
                    continue
                elif('conv1.1' in name):
                    total_elements_w = total_elements_w + m.num_elements * m.out_channels
                elif('conv2.1' in name):
                    total_elements_w = total_elements_w + m.num_elements * m.out_channels
                elif('shortcut.0.1' in name):
                    total_elements_w = total_elements_w + m.num_elements * m.out_channels
                elif('linear.1' in name):
                    total_elements_w = total_elements_w + m.num_elements
                elif('conv1.0' in name):
                    total_elements_a = total_elements_a + m.num_elements
                elif('conv2.0' in name):
                    total_elements_a = total_elements_a + m.num_elements
                elif('linear.0' in name):
                    total_elements_a = total_elements_a + m.num_elements
                elif('shortcut.0.0' in name):
                    total_elements_a = total_elements_a + m.num_elements

        loss_reg_w = 0
        loss_reg_a = 0
        bits_per_weight = 0
        bits_per_activation = 0
        for name, m in model.named_modules():
            if('conv1.1' in name):
                loss_reg_w = loss_reg_w + torch.sum((m.bits)**2) * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +  torch.sum(m.bits) * m.num_elements
                
            elif('conv2.1' in name):
                loss_reg_w = loss_reg_w + torch.sum((m.bits)**2) * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +  torch.sum(m.bits) * m.num_elements 
                
            elif('shortcut.0.1' in name):
                loss_reg_w = loss_reg_w + torch.sum((m.bits)**2) * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +  torch.sum(m.bits) * m.num_elements 

            elif('linear.1' in name):
                loss_reg_w = loss_reg_w + (m.bits)**2 * m.num_elements/total_elements_w
                bits_per_weight = bits_per_weight +   torch.clamp(m.bits, min = 1) * m.num_elements 
                # bits_per_weight = bits_per_weight +  torch.round(m.bits) * m.num_elements / total_elements_w
                if(m.bits < 1):
                    m.bits.requires_grad = False

            elif(name == 'conv1.0'):
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements 

            elif('conv1.0' in name):
                loss_reg_a = loss_reg_a + (m.bits)**2 * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements 
                if(m.bits < 1):
                    m.bits.requires_grad = False

            elif('conv2.0' in name):
                loss_reg_a = loss_reg_a + (m.bits)**2 * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements  
                if(m.bits < 1):
                    m.bits.requires_grad = False

            elif('linear.0' in name):
                loss_reg_a = loss_reg_a + (m.bits)**2 * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements  
                if(m.bits < 1):
                    m.bits.requires_grad = False

            elif('shortcut.0.0' in name):
                loss_reg_a = loss_reg_a + (m.bits)**2 * m.num_elements/total_elements_a
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits, min = 1) * m.num_elements 
                if(m.bits < 1):
                    m.bits.requires_grad = False
        bits_per_weight = bits_per_weight / total_elements_w
        bits_per_activation = bits_per_activation / total_elements_a
        loss_reg_w = args.hp.bit_penalty_w*loss_reg_w
        loss_reg_a = args.hp.bit_penalty_a*loss_reg_a
        loss = loss_ce + loss_reg_w + loss_reg_a
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        loss.backward()
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss, base_step + i)
            writer.add_scalar('loss/ce', loss_ce, base_step + i)
            writer.add_scalar('loss/reg_w', loss_reg_w, base_step + i)
            writer.add_scalar('loss/reg_a', loss_reg_a, base_step + i)
            writer.add_scalar('bits/bits_per_weight', bits_per_weight, base_step + i)
            writer.add_scalar('bits/bits_per_activation', bits_per_activation, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if('bits' in name):
            writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
        else:
            writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
        if param.grad is not None:
            writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    return bits_per_weight, bits_per_activation

def train_resnet20_split(train_loader, model_a, model_b, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model_a.train()
    model_b.train()

    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        #optimizer zero grad
        optimizer.zero_grad()
        if(i==0):
            for name, m in model_a.named_modules():
                    if ('conv1' in name and 'layer' not in name):
                        m.is_first.fill_(1)
                        break
            for name, m in model_b.named_modules():
                    if ('conv1' in name and 'layer' not in name):
                        m.is_first.fill_(1)
                        break
        # compute output of both models
        output_a = model_a(inputs)
        output_b = model_b(inputs)

        output = output_a * 4 + output_b
        
        loss_ce = criterion(output, targets)
        
        ################### Total Loss ####################
        loss = loss_ce 

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        loss.backward()
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss, base_step + i)
            writer.add_scalar('loss/ce', loss_ce, base_step + i)
            
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model_a.named_parameters():
        writer.add_histogram(name+"_A",param.data,epoch* len(train_loader) + i)
        if param.grad is not None:
            writer.add_histogram(name+"_A"+'/grad',param.grad.data,epoch* len(train_loader) + i)
    for name, param in model_b.named_parameters():
        writer.add_histogram(name+"_B",param.data,epoch* len(train_loader) + i)
        if param.grad is not None:
            writer.add_histogram(name+"_B"+'/grad',param.grad.data,epoch* len(train_loader) + i)
    return

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def train_resnet20_KD_bitops_ee(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer, target_bitops, confidence, bit_train_flag):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    
    #for getting intermediate activations for Knowledge distillation
    return_layers = {'conv1' : 'conv1',
                    'layer1.0.conv1' : 'layer1.0.conv1',
                    'layer1.0.conv2' : 'layer1.0.conv2',
                    'layer1.1.conv1' : 'layer1.1.conv1',
                    'layer1.1.conv2' : 'layer1.1.conv2',
                    'layer1.2.conv1' : 'layer1.2.conv1',
                    'layer1.2.conv2' : 'layer1.2.conv2',
                    'layer2.0.conv1' : 'layer2.0.conv1',
                    'layer2.0.conv2' : 'layer2.0.conv2',
                    'layer2.1.conv1' : 'layer2.1.conv1',
                    'layer2.1.conv2' : 'layer2.1.conv2',
                    'layer2.2.conv1' : 'layer2.2.conv1',
                    'layer2.2.conv2' : 'layer2.2.conv2',
                    'layer3.0.conv1' : 'layer3.0.conv1',
                    'layer3.0.conv2' : 'layer3.0.conv2',
                    'layer3.1.conv1' : 'layer3.1.conv1',
                    'layer3.1.conv2' : 'layer3.1.conv2',
                    'layer3.2.conv1' : 'layer3.2.conv1',
                    'layer3.2.conv2' : 'layer3.2.conv2'}
                    # 'linear' : 'linear'}
    intermidiate_getter_student = MidGetter(model, return_layers=return_layers, keep_output=True)
    intermediate_getter_teacher = MidGetter(teacher_model, return_layers=return_layers, keep_output=True)
    
    end = time.time()
    bitops_penalty = args.hp.bit_penalty_w 
    
    # cur_coeffs = torch.tensor([0.58, 0.68, 0.79, 0.89, 1])
    # cur_coeffs = torch.tensor([0.89, 0.79, 0.68, 0.58])
    # cur_coeffs = torch.tensor([0.8, 0.6, 0.4, 0.2])
    # cur_coeffs = torch.tensor([0.2, 0.4, 0.6, 0.8])
    cur_coeffs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    

    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        #optimizer zero grad
        optimizer.zero_grad()
        if(i==0):
            for name, m in model.named_modules():
                    if ('conv1' in name and 'layer' not in name):
                        #first layer activations to remain 8-bits and not trainable
                        m.bits_a.data.copy_(torch.ones(1) * 8)
                        m.bits_a.requires_grad = False
                        m.is_first.fill_(1)
                        break
            writer.add_graph(model, inputs)
        if epoch == 200 and i==0:
            for name, m in model.named_modules():
                if('conv' in name):
                    m.switch.fill_(0)
        
        # compute output and get intermediate activations of student and teacher
        intermediate_outputs_student, output_exit_student = intermidiate_getter_student(inputs)
        with torch.no_grad():
            intermediate_outputs_teacher, outputs_teacher = intermediate_getter_teacher(inputs)
        
        # with torch.no_grad():
        #     exit_probs = []
        #     for tt in range(0,len(output_exit_student)-1):
        #         prob = torch.nn.functional.softmax(output_exit_student[tt])
        #         prob = prob[range(targets.shape[0]), targets]
        #         exit_probs.append(prob)
        #     # cur_coeffs = torch.stack([(1 - (0.5*(exit_probs[i-1] + exit_probs[i-2])).pow(2)) if i >=2 else torch.cuda.FloatTensor(exit_probs[0].shape).fill_(0.5) for i in range (0,5) ]).unsqueeze(2)
        #     cur_coeffs = torch.stack([(exit_probs[i].pow(0.5)) for i in range (0,4)]).unsqueeze(2)
        
        loss_ce_last = (criterion(output_exit_student[-1], targets)).mean() 
        writer.add_scalar('loss_ce/exit'+str(len(output_exit_student)-1), loss_ce_last, base_step + i)
        loss_ce_last.backward(retain_graph=True)
        

        grad_dict_last = {}
        for name, m in model.named_parameters():
            # if('exit' not in name and 'layer3.2.conv2' not in name and 'layer3.2.bn2' not in name):
            if('exit' not in name):
                if m.grad is not None:
                    grad_dict_last[name] = m.grad.clone()
        
        optimizer.zero_grad()


        for tt in range(0,len(output_exit_student)-1):
            if tt == 0 :
                loss = criterion(output_exit_student[tt], targets)
                writer.add_scalar('loss_ce_unscale/exit'+str(tt), loss.mean(), base_step + i)
                loss = (cur_coeffs[tt] * loss).mean()
                # print(cur_coeffs[tt])
                loss_ce_exit = loss
                writer.add_scalar('loss_ce/exit'+str(tt), loss, base_step + i)
                writer.add_histogram('coeffs_ce/exit'+str(tt), cur_coeffs[tt], base_step + i)
            else:
                loss = criterion(output_exit_student[tt], targets) 
                writer.add_scalar('loss_ce_unscale/exit'+str(tt), loss.mean(), base_step + i)
                loss = (cur_coeffs[tt] * loss).mean() 
                loss_ce_exit += loss
                writer.add_scalar('loss_ce/exit'+str(tt), loss, base_step + i)
                writer.add_histogram('coeffs_ce/exit'+str(tt), cur_coeffs[tt], base_step + i)
                
        
        loss_ce_exit.backward(retain_graph = True)
        
        grad_dict_exit = {}
        for name, m in model.named_parameters():
            if m.grad is not None:
                grad_dict_exit[name] = m.grad.clone()
        
        optimizer.zero_grad()

        losses.update((loss_ce_last+loss_ce_exit).item(), inputs.size(0))
        

        writer.add_scalar('loss_ce/total'+str(tt), loss_ce_last+loss_ce_exit, base_step + i)

        grad_dict_new = copy.deepcopy(grad_dict_exit)
        with torch.no_grad():
            for k in grad_dict_new.keys():
                if('exit' not in k):
                    grad_exit = grad_dict_exit.pop(k)
                    grad_last = grad_dict_last.pop(k)
                    mask = grad_exit.sign().eq(grad_last.sign()) 
                    grad_exit = grad_exit * mask
                    # similarity = torch.dot(grad_exit.flatten(), grad_last.flatten())  
                    # if similarity < 0:
                    #     grad_dict_exit[k] = grad_last + (grad_exit - (similarity * grad_last) / (grad_last.norm()**2 + 1e-8))
                    # else:
                    #     grad_dict_exit[k] = grad_exit + grad_last
                    # if ('weight' in k):
                    similarity = torch.nn.functional.cosine_similarity(grad_exit.flatten(), grad_last.flatten(), dim = 0, eps = 1e-11)
                    writer.add_scalar("similarity_cosine_4b/"+k, similarity, base_step+i)
                    # if (similarity < 0.2):
                    # grad_dict_exit[k] = grad_last + grad_exit * grad_exit.sign().eq(grad_last.sign()) 
                    # else:
                    grad_dict_exit[k] = grad_exit + grad_last

        with torch.no_grad():
            for name, m in model.named_parameters():
                if m.grad is not None:
                    m.grad.data.zero_()
                    m.grad.data.copy_(grad_dict_exit[name])
        
        
       
        output = output_exit_student[len(output_exit_student)-1]
        output_teacher = outputs_teacher[len(output_exit_student)-1] 
        
        if (i == 0):
            #get elements in each tensor which are quantized (for bits/w and bits/a calculation)
            total_elements_w = 0
            total_elements_a = 0
            for name, m in model.named_modules():
                if ('conv1' in name and 'layer' not in name and 'exit' not in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                    #first layer activations to remain 8-bits and not trainable
                    m.bits_a.data.copy_(torch.ones(1) * 8)
                    m.bits_a.requires_grad = False
                    continue
                elif(('conv' in name and 'layer' in name) or 'linear' in name):
                    # print(name)
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
        ################### Bitops Loss ####################        
        bitops = torch.cuda.DoubleTensor([0])
        bits_per_weight = torch.cuda.DoubleTensor([0])
        bits_per_activation = torch.cuda.DoubleTensor([0])
        for name, m in model.named_modules():
            if(('conv' in name )):
                bits_per_weight = bits_per_weight +  torch.mean(m.bits_w) * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.mean(m.bits_a) * m.num_elements_a / total_elements_a
                if((m.bits_a < 2).int().sum() > 0):
                    m.bits_a.requires_grad = False
                if((m.bits_w < 2).int().sum() > 0):
                    m.bits_w.requires_grad = False
                bits_w = (torch.mean(m.bits_w)).detach() - (torch.sum(m.bits_w)).detach() + (torch.sum(m.bits_w)) 
                bits_a = (torch.mean(m.bits_a)).detach() - (torch.sum(m.bits_a)).detach() + (torch.sum(m.bits_a)) 
                # bitops = bitops +  torch.mean(m.bits_a) * torch.mean(m.bits_w) * m.bitops
                bitops = bitops +  bits_a * bits_w * m.bitops
            if(('linear' in name)):
                bits_per_weight = bits_per_weight +  torch.mean(m.bits_w) * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   (m.bits_a) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                if((m.bits_w < 2).int().sum() > 0):
                    m.bits_w.requires_grad = False
                bitops = bitops + (m.bits_a) * (torch.mean(m.bits_w)) * m.bitops 
        bitops = bitops / 1000000000   
        loss_reg = bitops_penalty * torch.abs(bitops - target_bitops) 
        # print(total_elements_w)
        ################### KD Loss ####################
        loss_kd_inter = 0
        loss_kd_sce = 0
        alpha_kd = 0.01
        T = 10
        alpha_kd_sce = 0.01
        # for name, _ in intermediate_outputs_teacher.items():
        #     teacher_tensor = intermediate_outputs_teacher[name] / torch.linalg.matrix_norm(intermediate_outputs_teacher[name], keepdim = True)
        #     student_tensor = intermediate_outputs_student[name] / torch.linalg.matrix_norm(intermediate_outputs_student[name], keepdim = True)
        #     loss_kd_inter =+ alpha_kd * torch.nn.functional.mse_loss(student_tensor, teacher_tensor)
        # loss_kd_sce = alpha_kd_sce * torch.nn.functional.kl_div(torch.nn.functional.log_softmax(output/T, dim=1),
        #                      torch.nn.functional.softmax(output_teacher/T, dim=1)) * T * T
        
        ################### Total Loss other than Crossentropy####################
        loss = loss_reg + loss_kd_inter + loss_kd_sce

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        if (bit_train_flag):
            loss.backward()
            # for name, m in model.named_parameters():
            #     if('bits_w' in name):
            #         print(m.grad)
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss, base_step + i)
            writer.add_scalar('loss/reg_bitops', loss_reg, base_step + i)
            writer.add_scalar('loss/reg_KD_inter', loss_kd_inter, base_step + i)
            writer.add_scalar('loss/reg_KD_sce', loss_kd_sce, base_step + i)
            writer.add_scalar('bits/bits_per_weight', bits_per_weight, base_step + i)
            writer.add_scalar('bits/bits_per_activation', bits_per_activation, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        if epoch == 200 and i==0:
            for name, m in model.named_modules():
                if('conv' in name):
                    m.switch.fill_(0)
        
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if('bits' in name):
            writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
        # elif('beta_w' in name and 'conv' in name):
        #     writer.add_scalars(f'threshold/'+name, {
        #         '-8': param.data[0],
        #         '-7': param.data[1],
        #         '-6': param.data[2],
        #         '-5': param.data[3],
        #         '-4': param.data[4],
        #         '-3': param.data[5],
        #         '-2': param.data[6],
        #         '-1': param.data[7],
        #         '0': param.data[8],
        #         '1': param.data[9],
        #         '2': param.data[10],
        #         '3': param.data[11],
        #         '4': param.data[12],
        #         '5': param.data[13],
        #         '6': param.data[14],
        #     }, epoch* len(train_loader) + i)
        else:
            writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
        if param.grad is not None:
            writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    
    return bits_per_weight, bits_per_activation, bitops

def train_resnet56_KD_bitops_ee(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer, target_bitops, bit_train_flag):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    
    #for getting intermediate activations for Knowledge distillation
    
    
    end = time.time()
    bitops_penalty = args.hp.bit_penalty_w 
    
    # cur_coeffs = torch.tensor([0.58, 0.68, 0.79, 0.89, 1])
    # cur_coeffs = torch.tensor([0.89, 0.79, 0.68, 0.58])
    # cur_coeffs = torch.tensor([0.8, 0.6, 0.4, 0.2])
    # cur_coeffs = torch.tensor([0.2, 0.4, 0.6, 0.8])
    cur_coeffs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    

    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        #optimizer zero grad
        optimizer.zero_grad()

        if(i==0):
            for name, m in model.named_modules():
                    if ('conv1' in name and 'layer' not in name):
                        #first layer activations to remain 8-bits and not trainable
                        m.bits_a.data.copy_(torch.ones(1) * 8)
                        m.bits_a.requires_grad = False
                        m.is_first.fill_(1)
                        break
            # writer.add_graph(model, inputs)
        if epoch == 200 and i==0:
            for name, m in model.named_modules():
                if('conv' in name):
                    m.switch.fill_(0)
        
        # compute output and get intermediate activations of student and teacher
        output_exit_student = model(inputs)
        with torch.no_grad():
            if teacher_model is not None:
                outputs_teacher = teacher_model(inputs)
        
        # with torch.no_grad():
        #     exit_probs = []
        #     for tt in range(0,len(output_exit_student)-1):
        #         prob = torch.nn.functional.softmax(output_exit_student[tt])
        #         prob = prob[range(targets.shape[0]), targets]
        #         exit_probs.append(prob)
        #     # cur_coeffs = torch.stack([(1 - (0.5*(exit_probs[i-1] + exit_probs[i-2])).pow(2)) if i >=2 else torch.cuda.FloatTensor(exit_probs[0].shape).fill_(0.5) for i in range (0,5) ]).unsqueeze(2)
        #     cur_coeffs = torch.stack([(exit_probs[i].pow(0.5)) for i in range (0,4)]).unsqueeze(2)
        
        loss_ce_last = (criterion(output_exit_student[-1], targets)).mean() 
        writer.add_scalar('loss_ce/exit'+str(len(output_exit_student)-1), loss_ce_last, base_step + i)
        loss_ce_last.backward(retain_graph=True)
        
        grad_dict_last = {}
        for name, m in model.named_parameters():
            if m.grad is not None:
                grad_dict_last[name] = m.grad.clone()

        optimizer.zero_grad()
        for tt in range(0,len(output_exit_student)-1):
            if tt == 0 :
                loss = criterion(output_exit_student[tt], targets)
                writer.add_scalar('loss_ce_unscale/exit'+str(tt), loss.mean(), base_step + i)
                loss = (cur_coeffs[tt] * loss).mean()
                loss_ce_exit = loss
                writer.add_scalar('loss_ce/exit'+str(tt), loss, base_step + i)
                writer.add_histogram('coeffs_ce/exit'+str(tt), cur_coeffs[tt], base_step + i)
            else:
                loss = criterion(output_exit_student[tt], targets) 
                writer.add_scalar('loss_ce_unscale/exit'+str(tt), loss.mean(), base_step + i)
                loss = (cur_coeffs[tt] * loss).mean() 
                loss_ce_exit += loss
                writer.add_scalar('loss_ce/exit'+str(tt), loss, base_step + i)
                writer.add_histogram('coeffs_ce/exit'+str(tt), cur_coeffs[tt], base_step + i)
                
        
        loss_ce_exit.backward(retain_graph = True)
        
        grad_dict_exit = {}
        for name, m in model.named_parameters():
            if m.grad is not None:
                grad_dict_exit[name] = m.grad.clone()
        
        optimizer.zero_grad()

        

        
        

        writer.add_scalar('loss_ce/total'+str(tt), loss_ce_last+loss_ce_exit, base_step + i)

        grad_dict_new = copy.deepcopy(grad_dict_exit)
        with torch.no_grad():
            for k in grad_dict_new.keys():
                if('exit' not in k):
                    grad_exit = grad_dict_exit.pop(k)
                    grad_last = grad_dict_last.pop(k)
                    similarity = torch.dot(grad_exit.flatten(), grad_last.flatten())  
                    # if similarity < 0:
                        # grad_dict_exit[k] = grad_last + (grad_exit - (similarity * grad_last) / (grad_last.norm()**2 + 1e-8))
                    # else:
                        # grad_dict_exit[k] = grad_exit + grad_last
                    grad_dict_exit[k] = grad_last + grad_exit * grad_exit.sign().eq(grad_last.sign()) 
                    # sign_similarity = (grad_exit.sign().eq(grad_last.sign())).sum()/grad_exit.nelement()
                    # writer.add_scalar('similarity/'+str(k), sign_similarity, base_step + i)
                    # grad_dict_exit[k] = grad_exit + grad_last
                
        grad_dict_new = copy.deepcopy(grad_dict_last)
        with torch.no_grad():
            for k in grad_dict_new.keys():
                if m.grad is not None:
                    m.grad.data.zero_()
                    m.grad.data.copy_(grad_dict_exit[name])

        
        
       
        output = output_exit_student[len(output_exit_student)-1]
        
        if (i == 0):
            #get elements in each tensor which are quantized (for bits/w and bits/a calculation)
            total_elements_w = 0
            total_elements_a = 0
            for name, m in model.named_modules():
                if ('conv1' in name and 'layer' not in name and 'exit' not in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                    #first layer activations to remain 8-bits and not trainable
                    m.bits_a.data.copy_(torch.ones(1) * 8)
                    m.bits_a.requires_grad = False
                    continue
                elif(('conv' in name and 'layer' in name) or 'linear' in name):
                    # print(name)
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
        ################### Bitops Loss ####################        
        bitops = torch.cuda.DoubleTensor([0])
        bits_per_weight = torch.cuda.DoubleTensor([0])
        bits_per_activation = torch.cuda.DoubleTensor([0])
        for name, m in model.named_modules():
            if (isinstance(m, my_nn.Conv2dLSQ) or isinstance(m, my_nn.LinearLSQ)):
                bits_a = m.bits_a
                bits_w = m.bits_w
                if(bits_a < 2):
                    bits_a = torch.clamp(bits_a, min = 2)
                if(bits_w.le(2).sum() > 0):
                    bits_w = torch.clamp(bits_w, min = 2)
                bits_w = torch.mean(bits_w)
                bits_per_weight = bits_per_weight +  bits_w * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   bits_a * m.num_elements_a / total_elements_a
                bitops_layer = bits_a * bits_w * m.bitops
                bitops = bitops +  bitops_layer
                # print(name + " : " + str(float(bitops_layer/ 1000000000)) + ", bits/w : " + str(float(bits_w)) + ", bits/a : " + str(float(bits_a))) 
        # print("DONE")
        bitops = bitops / 1000000000   
        loss_reg = bitops_penalty * torch.abs(bitops - target_bitops) 
        # print(total_elements_w)
        ################### KD Loss ####################
        loss_kd_inter = 0
        loss_kd_sce = 0
        alpha_kd = 0.01
        T = 10
        alpha_kd_sce = 0.01
        # for name, _ in intermediate_outputs_teacher.items():
        #     teacher_tensor = intermediate_outputs_teacher[name] / torch.linalg.matrix_norm(intermediate_outputs_teacher[name], keepdim = True)
        #     student_tensor = intermediate_outputs_student[name] / torch.linalg.matrix_norm(intermediate_outputs_student[name], keepdim = True)
        #     loss_kd_inter =+ alpha_kd * torch.nn.functional.mse_loss(student_tensor, teacher_tensor)
        # loss_kd_sce = alpha_kd_sce * torch.nn.functional.kl_div(torch.nn.functional.log_softmax(output/T, dim=1),
        #                      torch.nn.functional.softmax(output_teacher/T, dim=1)) * T * T
        
        ################### Total Loss other than Crossentropy####################
        loss = loss_reg + loss_kd_inter + loss_kd_sce

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update((loss_ce_last+loss_ce_exit+loss).item(), inputs.size(0))

        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        if (bit_train_flag):
            loss.backward()
            # for name, m in model.named_parameters():
            #     if('bits_w' in name):
            #         print(m.grad)
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss, base_step + i)
            writer.add_scalar('loss/reg_bitops', loss_reg, base_step + i)
            writer.add_scalar('loss/reg_KD_inter', loss_kd_inter, base_step + i)
            writer.add_scalar('loss/reg_KD_sce', loss_kd_sce, base_step + i)
            writer.add_scalar('bits/bits_per_weight', bits_per_weight, base_step + i)
            writer.add_scalar('bits/bits_per_activation', bits_per_activation, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        if epoch == 200 and i==0:
            for name, m in model.named_modules():
                if('conv' in name):
                    m.switch.fill_(0)
        
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if('bits' in name):
            writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
        # elif('beta_w' in name and 'conv' in name):
        #     writer.add_scalars(f'threshold/'+name, {
        #         '-8': param.data[0],
        #         '-7': param.data[1],
        #         '-6': param.data[2],
        #         '-5': param.data[3],
        #         '-4': param.data[4],
        #         '-3': param.data[5],
        #         '-2': param.data[6],
        #         '-1': param.data[7],
        #         '0': param.data[8],
        #         '1': param.data[9],
        #         '2': param.data[10],
        #         '3': param.data[11],
        #         '4': param.data[12],
        #         '5': param.data[13],
        #         '6': param.data[14],
        #     }, epoch* len(train_loader) + i)
        else:
            writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
        if param.grad is not None:
            writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    
    return bits_per_weight, bits_per_activation, bitops

def train_resnet20_KD_bitops(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer, target_bitops):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()

    #for getting intermediate activations for Knowledge distillation
    return_layers = {'conv1' : 'conv1',
                    'layer1.0.conv1' : 'layer1.0.conv1',
                    'layer1.0.conv2' : 'layer1.0.conv2',
                    'layer1.1.conv1' : 'layer1.1.conv1',
                    'layer1.1.conv2' : 'layer1.1.conv2',
                    'layer1.2.conv1' : 'layer1.2.conv1',
                    'layer1.2.conv2' : 'layer1.2.conv2',
                    'layer2.0.conv1' : 'layer2.0.conv1',
                    'layer2.0.conv2' : 'layer2.0.conv2',
                    'layer2.1.conv1' : 'layer2.1.conv1',
                    'layer2.1.conv2' : 'layer2.1.conv2',
                    'layer2.2.conv1' : 'layer2.2.conv1',
                    'layer2.2.conv2' : 'layer2.2.conv2',
                    'layer3.0.conv1' : 'layer3.0.conv1',
                    'layer3.0.conv2' : 'layer3.0.conv2',
                    'layer3.1.conv1' : 'layer3.1.conv1',
                    'layer3.1.conv2' : 'layer3.1.conv2',
                    'layer3.2.conv1' : 'layer3.2.conv1',
                    'layer3.2.conv2' : 'layer3.2.conv2',
                    'linear' : 'linear'}
    intermidiate_getter_student = MidGetter(model, return_layers=return_layers, keep_output=True)
    intermediate_getter_teacher = MidGetter(teacher_model, return_layers=return_layers, keep_output=True)
    
    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        #optimizer zero grad
        optimizer.zero_grad()
        if(i==0):
            for name, m in model.named_modules():
                    if ('conv1' in name and 'layer' not in name):
                        #first layer activations to remain 8-bits and not trainable
                        m.bits_a.data.copy_(torch.ones(1) * 8)
                        m.bits_a.requires_grad = False
                        m.is_first.fill_(1)
                        break
        
        # compute output and get intermediate activations of student and teacher
        intermediate_outputs_student, output = intermidiate_getter_student(inputs)
        with torch.no_grad():
            intermediate_outputs_teacher, output_teacher = intermediate_getter_teacher(inputs)
        
        loss_ce = criterion(output, targets)
        
        
        if (i == 0):
            #get elements in each tensor which are quantized (for bits/w and bits/a calculation)
            total_elements_w = 0
            total_elements_a = 0
            for name, m in model.named_modules():
                if ('conv1' in name and 'layer' not in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                    #first layer activations to remain 8-bits and not trainable
                    m.bits_a.data.copy_(torch.ones(1) * 8)
                    m.bits_a.requires_grad = False
                    continue
                elif('conv1' in name and 'layer' in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                elif('conv2' in name and 'layer' in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                elif('shortcut.0' in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                elif('linear' in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
        ################### Bitops Loss ####################        
        bitops = 0
        bits_per_weight = 0
        bits_per_activation = 0
        for name, m in model.named_modules():
            if ('conv1' in name and 'layer' not in name):
                bits_per_weight = bits_per_weight +  torch.round(m.bits_w) * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.round(m.bits_a) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                if(m.bits_w < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops

            elif('conv1' in name and 'layer' in name):
                bits_per_weight = bits_per_weight +  torch.round(m.bits_w) * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.round(m.bits_a) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                if(m.bits_w < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops

            elif('conv2' in name and 'layer' in name):
                bits_per_weight = bits_per_weight +  torch.round(m.bits_w) * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.round(m.bits_a) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                if(m.bits_w < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops
            
            elif('shortcut.0' in name):
                bits_per_weight = bits_per_weight +  torch.round(m.bits_w) * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.round(m.bits_a) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                if(m.bits_w < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops
            
            elif('linear' in name):
                bits_per_weight = bits_per_weight +  torch.round(m.bits_w) * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.round(m.bits_a) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                if(m.bits_w < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops
        
        bitops = bitops / 1000000000   
        loss_reg = args.hp.bit_penalty_w * torch.abs(bitops - target_bitops) 
        # print(total_elements_w)
        ################### KD Loss ####################
        loss_kd_inter = 0
        alpha_kd = 0.01
        T = 10
        alpha_kd_sce = 0.01
        for name, _ in intermediate_outputs_teacher.items():
            teacher_tensor = intermediate_outputs_teacher[name] / torch.linalg.matrix_norm(intermediate_outputs_teacher[name], keepdim = True)
            student_tensor = intermediate_outputs_student[name] / torch.linalg.matrix_norm(intermediate_outputs_student[name], keepdim = True)
            loss_kd_inter =+ alpha_kd * torch.nn.functional.mse_loss(student_tensor, teacher_tensor)
        loss_kd_sce = alpha_kd_sce * torch.nn.functional.kl_div(torch.nn.functional.log_softmax(output/T, dim=1),
                             torch.nn.functional.softmax(output_teacher/T, dim=1)) * T * T
        
        ################### Total Loss ####################
        loss = loss_ce + loss_reg + loss_kd_inter + loss_kd_sce

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        loss.backward()
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss, base_step + i)
            writer.add_scalar('loss/ce', loss_ce, base_step + i)
            writer.add_scalar('loss/reg_bitops', loss_reg, base_step + i)
            writer.add_scalar('loss/reg_KD_inter', loss_kd_inter, base_step + i)
            writer.add_scalar('loss/reg_KD_sce', loss_kd_sce, base_step + i)
            writer.add_scalar('bits/bits_per_weight', bits_per_weight, base_step + i)
            writer.add_scalar('bits/bits_per_activation', bits_per_activation, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if('bits' in name):
            writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
        else:
            writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
        if param.grad is not None:
            writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    return bits_per_weight, bits_per_activation, bitops

def train_resnet20_kernelwise_KD_bitops(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer, target_bitops):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()

    #for getting intermediate activations for Knowledge distillation
    return_layers = {'conv1' : 'conv1',
                    'layer1.0.conv1' : 'layer1.0.conv1',
                    'layer1.0.conv2' : 'layer1.0.conv2',
                    'layer1.1.conv1' : 'layer1.1.conv1',
                    'layer1.1.conv2' : 'layer1.1.conv2',
                    'layer1.2.conv1' : 'layer1.2.conv1',
                    'layer1.2.conv2' : 'layer1.2.conv2',
                    'layer2.0.conv1' : 'layer2.0.conv1',
                    'layer2.0.conv2' : 'layer2.0.conv2',
                    'layer2.1.conv1' : 'layer2.1.conv1',
                    'layer2.1.conv2' : 'layer2.1.conv2',
                    'layer2.2.conv1' : 'layer2.2.conv1',
                    'layer2.2.conv2' : 'layer2.2.conv2',
                    'layer3.0.conv1' : 'layer3.0.conv1',
                    'layer3.0.conv2' : 'layer3.0.conv2',
                    'layer3.1.conv1' : 'layer3.1.conv1',
                    'layer3.1.conv2' : 'layer3.1.conv2',
                    'layer3.2.conv1' : 'layer3.2.conv1',
                    'layer3.2.conv2' : 'layer3.2.conv2',
                    'linear' : 'linear'}
    intermidiate_getter_student = MidGetter(model, return_layers=return_layers, keep_output=True)
    intermediate_getter_teacher = MidGetter(teacher_model, return_layers=return_layers, keep_output=True)
    
    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        #optimizer zero grad
        optimizer.zero_grad()
        if(i==0):
            for name, m in model.named_modules():
                    if ('conv1' in name and 'layer' not in name):
                        #first layer activations to remain 8-bits and not trainable
                        m.bits_a.data.copy_(torch.ones(1) * 8)
                        m.bits_a.requires_grad = False
                        m.is_first.fill_(1)
                        break
        
        # compute output and get intermediate activations of student and teacher
        intermediate_outputs_student, output = intermidiate_getter_student(inputs)
        with torch.no_grad():
            intermediate_outputs_teacher, output_teacher = intermediate_getter_teacher(inputs)
        
        loss_ce = criterion(output, targets)
        
        
        if (i == 0):
            #get elements in each tensor which are quantized (for bits/w and bits/a calculation)
            total_elements_w = 0
            total_elements_a = 0
            for name, m in model.named_modules():
                if ('conv1' in name and 'layer' not in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                    #first layer activations to remain 8-bits and not trainable
                    m.bits_a.data.copy_(torch.ones(1) * 8)
                    m.bits_a.requires_grad = False
                    continue
                elif('conv1' in name and 'layer' in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                elif('conv2' in name and 'layer' in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                elif('shortcut.0' in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                elif('linear' in name):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
        ################### Bitops Loss ####################        
        bitops = 0
        bits_per_weight = 0
        bits_per_activation = 0
        for name, m in model.named_modules():
            if ('conv1' in name and 'layer' not in name):
                bits_per_weight = bits_per_weight +  torch.clamp(m.bits_w, min = 2).mean() * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits_a, min = 1) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + (m.bits_a) * (m.bits_w).mean() * m.bitops
                
            elif('conv1' in name and 'layer' in name):
                bits_per_weight = bits_per_weight +  torch.clamp(m.bits_w, min = 2).mean() * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits_a, min = 1) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + (m.bits_a) * (m.bits_w).mean() * m.bitops

            elif('conv2' in name and 'layer' in name):
                bits_per_weight = bits_per_weight +  torch.clamp(m.bits_w, min = 2).mean() * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits_a, min = 1) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + (m.bits_a) * (m.bits_w).mean() * m.bitops
            
            elif('shortcut.0' in name):
                bits_per_weight = bits_per_weight +  torch.clamp(m.bits_w, min = 2).mean() * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits_a, min = 1) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + (m.bits_a) * (m.bits_w).mean() * m.bitops
            
            elif('linear' in name):
                bits_per_weight = bits_per_weight +  torch.clamp(m.bits_w, min = 2).mean() * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   torch.clamp(m.bits_a, min = 1) * m.num_elements_a / total_elements_a
                if(m.bits_a < 2):
                    m.bits_a.requires_grad = False
                bitops = bitops + (m.bits_a) * (m.bits_w).mean() * m.bitops
        
        bitops = bitops / 1000000000   
        loss_reg = args.hp.bit_penalty_w * torch.abs(bitops - target_bitops) 
        # print(total_elements_w)
        ################### KD Loss ####################
        loss_kd_inter = 0
        alpha_kd = 0.01
        T = 10
        alpha_kd_sce = 0.01
        for name, _ in intermediate_outputs_teacher.items():
            teacher_tensor = intermediate_outputs_teacher[name] / torch.linalg.matrix_norm(intermediate_outputs_teacher[name], keepdim = True)
            student_tensor = intermediate_outputs_student[name] / torch.linalg.matrix_norm(intermediate_outputs_student[name], keepdim = True)
            loss_kd_inter =+ alpha_kd * torch.nn.functional.mse_loss(student_tensor, teacher_tensor)
        loss_kd_sce = alpha_kd_sce * torch.nn.functional.kl_div(torch.nn.functional.log_softmax(output/T, dim=1),
                             torch.nn.functional.softmax(output_teacher/T, dim=1)) * T * T
        
        ################### Total Loss ####################
        loss = loss_ce + loss_reg + loss_kd_inter + loss_kd_sce

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        loss.backward()
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss, base_step + i)
            writer.add_scalar('loss/ce', loss_ce, base_step + i)
            writer.add_scalar('loss/reg_bitops', loss_reg, base_step + i)
            writer.add_scalar('loss/reg_KD_inter', loss_kd_inter, base_step + i)
            writer.add_scalar('loss/reg_KD_sce', loss_kd_sce, base_step + i)
            writer.add_scalar('bits/bits_per_weight', bits_per_weight, base_step + i)
            writer.add_scalar('bits/bits_per_activation', bits_per_activation, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if('bits' in name):
            writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
        else:
            writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
        if param.grad is not None:
            writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    return bits_per_weight, bits_per_activation, bitops

def round_sigmoid(x, T):
    y = x.round()
    y_grad = torch.sigmoid((x - 0.5*(x.floor()+x.ceil()))/T) + x.floor()
    return y.detach() - y_grad.detach() + y_grad
    # return x
def round_ste(x, T):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad




def train_resnet18(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer, target_bitops, bit_train_flag, switch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    
    end = time.time()
    bitops_penalty = args.hp.gamma_bop 

    base_step = epoch * args.batch_num

    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        
        if teacher_model is not None: 
            with torch.no_grad():
                targets_teacher = torch.nn.functional.softmax(teacher_model(inputs), dim = 1)
        
                        
        #optimizer zero grad
        optimizer.zero_grad()

        
        if(i==0):
            for name, m in model.named_modules():
                    if ('module.conv1' in name):
                        #first layer activations and weights to remain 8-bits and not trainable
                        m.bits_a.data.copy_(torch.ones(1) * 8)
                        m.bits_a.requires_grad = False
                        break
        
        
        # compute output and get intermediate activations of student and teacher
        output_exit_student = model(inputs)
        
        
        if teacher_model is not None:
            loss_ce_last = criterion(output_exit_student[-1], targets_teacher) 
        else:
            loss_ce_last = criterion(output_exit_student[-1], targets) 
        # if writer is not None:
            # writer.add_scalar('loss_ce/exit'+str(len(output_exit_student)-1), loss_ce_last, base_step + i)
        loss_ce_last.backward(retain_graph=True)
        
        grad_dict_last = {}
        for name, m in model.named_parameters():
            if m.grad is not None:
                grad_dict_last[name] = m.grad.clone()
        
        optimizer.zero_grad()
        for tt in range(0,len(output_exit_student)-1):
            if teacher_model is not None:
                loss = criterion(output_exit_student[tt], targets_teacher)
            else:
                loss = criterion(output_exit_student[tt], targets)
            if tt == 0:
                loss_ce_exit =  loss
            else: 
                loss_ce_exit += loss
            if writer is not None:
                writer.add_scalar('loss_ce/exit'+str(tt), loss, base_step + i)
                
        loss_ce_exit.backward(retain_graph = True)
        
        grad_dict_exit = {}
        for name, m in model.named_parameters():
            if m.grad is not None:
                grad_dict_exit[name] = m.grad.clone()
        
        
        optimizer.zero_grad()

        
        if writer is not None:
            writer.add_scalar('loss_ce/total'+str(tt), loss_ce_last+loss_ce_exit, base_step + i)

        grad_dict_new = copy.deepcopy(grad_dict_exit)
        with torch.no_grad():
            for k in grad_dict_new.keys():
                if('exit' not in k):
                    grad_exit = grad_dict_exit.pop(k)
                    grad_last = grad_dict_last.pop(k)
                    grad_dict_exit[k] = grad_last + grad_exit * grad_exit.sign().eq(grad_last.sign()) 
                    
        grad_dict_new = copy.deepcopy(grad_dict_last)
        
        with torch.no_grad():
            for name, m in model.named_parameters():
                if m.grad is not None:
                    m.grad.data.zero_()
                    m.grad.data.copy_(grad_dict_exit[name])
        
        
        if (i == 0):
            #get elements in each tensor which are quantized (for bits/w and bits/a calculation)
            total_elements_w = 0
            total_elements_a = 0
            for name, m in model.named_modules():
                if (isinstance(m, my_nn.Conv2dLSQ) or isinstance(m, my_nn.LinearLSQ)):
                    total_elements_a = total_elements_a + m.num_elements_a
                    total_elements_w = total_elements_w + m.num_elements_w
                    if ('module.features.0.0' in name):
                        #first layer activations to remain 8-bits and not trainable
                        m.bits_a.data.copy_(torch.ones(1) * 8)
                        m.bits_a.requires_grad = False
                        continue

        # ################### Bitops Loss ####################        
        bitops = torch.cuda.DoubleTensor([0])
        bits_per_weight = torch.cuda.DoubleTensor([0])
        bits_per_activation = torch.cuda.DoubleTensor([0])
        for name, m in model.named_modules():
            if (isinstance(m, my_nn.Conv2dLSQ) or isinstance(m, my_nn.LinearLSQ)):
                bits_a = m.bits_a
                bits_w = m.bits_w
                if(bits_a < 2):
                    bits_a = torch.clamp(bits_a, min = 2)
                if(bits_w.le(2).sum() > 0):
                    bits_w = torch.clamp(bits_w, min = 2)
                bits_w = torch.mean(bits_w)
                bits_per_weight = bits_per_weight +  bits_w * m.num_elements_w / total_elements_w
                bits_per_activation = bits_per_activation +   bits_a * m.num_elements_a / total_elements_a
                
               
                bitops = bitops +  bits_a * bits_w * m.bitops 
        bitops = bitops / 1000000000   
        loss_reg = bitops_penalty * torch.abs(bitops - target_bitops) 
        
        
        ################### Total Loss other than Crossentropy####################
        loss = loss_reg

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_exit_student[-1], targets, topk=(1, 5))
        losses.update((loss+loss_ce_last).item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        if (bit_train_flag):
            loss.backward(retain_graph = True)
        
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss, base_step + i)
            writer.add_scalar('loss/reg_bitops', loss_reg, base_step + i)
            writer.add_scalar('bits/bits_per_weight', bits_per_weight, base_step + i)
            writer.add_scalar('bits/bits_per_activation', bits_per_activation, base_step + i)
            writer.add_scalar('bits/bitops', bitops, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
        # for name, param in model.named_parameters():
        #     if writer is not None:
        #         if('bits' in name):
        #             writer.add_scalar('bits/'+name,param.data,epoch* len(train_loader) + i)
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if writer is not None:
            if('bits' in name):
                # writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
                continue
            else:
                writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
            if param.grad is not None:
                writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    
    return bits_per_weight, bits_per_activation, bitops

def train_resnet18_first(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    
    end = time.time()

    base_step = epoch * args.batch_num

    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        
        if teacher_model is not None: 
            with torch.no_grad():
                targets_teacher = torch.nn.functional.softmax(teacher_model(inputs), dim = 1)
        
                        
        #optimizer zero grad
        optimizer.zero_grad()

        
        if(i==0):
            for name, m in model.named_modules():
                    if ('module.conv1' in name):
                        #first layer activations and weights to remain 8-bits and not trainable
                        m.bits_a.data.copy_(torch.ones(1) * 8)
                        m.bits_a.requires_grad = False
                        break
        
        # compute output and get intermediate activations of student and teacher
        output_exit_student = model(inputs)
        
        
        
        optimizer.zero_grad()
        for tt in range(0,len(output_exit_student)):
            if teacher_model is not None:
                loss = criterion(output_exit_student[tt], targets_teacher)
            else:
                loss = criterion(output_exit_student[tt], targets)
            if tt == 0:
                loss_ce_exit =  loss
            else: 
                loss_ce_exit += loss
            if writer is not None:
                writer.add_scalar('loss_ce/exit'+str(tt), loss, base_step + i)
                
        loss_ce_exit.backward(retain_graph = True)
        

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_exit_student[-1], targets, topk=(1, 5))
        losses.update((loss_ce_exit).item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.step()
       
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss_ce_exit, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
        
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if writer is not None:
            if('bits' in name):
                writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
            else:
                writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
            if param.grad is not None:
                writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    
    return 



def train_conventional(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    
    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        
        #optimizer zero grad
        optimizer.zero_grad()
        
        if(i==0):
            for name, m in model.named_modules():
                    if ('conv1' in name and 'layer' not in name):
                        #first layer activations to remain 8-bits and not trainable
                        m.bits_a.data.copy_(torch.ones(1) * 8)
                        m.bits_a.requires_grad = False
                        m.is_first.fill_(1)
                        break
        
        # compute output and get intermediate activations of student and teacher
        output_exit_student = model(inputs)
        # print(len(output_exit_student))
        loss_ce_exit = 0
        # for tt in range(0,len(output_exit_student)):
        #     loss = criterion(output_exit_student[tt], targets).mean()
        #     loss_ce_exit += loss
        #     if writer is not None:
        #         writer.add_scalar('loss_ce/exit'+str(tt), loss, base_step + i)
        loss = criterion(output_exit_student[-1], targets).mean()
        loss_ce_exit += loss
        loss_ce_exit.backward(retain_graph = True)
        
        # for name, m in model.named_parameters():
        #     if ('module.features.9.exit_conv.0.weight' in name):
        #         print(m.grad)
        
        if writer is not None:
            writer.add_scalar('loss_ce/total', loss_ce_exit, base_step + i)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_exit_student[-1], targets, topk=(1, 5))
        losses.update((loss_ce_exit).item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
        for name, param in model.named_parameters():
            if writer is not None:
                if ('conv' in name and 'weight' in name):
                    if param.grad is not None:
                        for idxx in range(param.grad.data.shape[0]):
                            writer.add_scalar('variance_kernelwise_'+name+'/'+str(idxx),torch.var(param.grad.data[idxx], dim = (0,1,2)),epoch* len(train_loader) + i)
                        writer.add_scalar('variance_layerwise/'+name,torch.var(param.grad.data, dim = (0,1,2,3)),epoch* len(train_loader) + i)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if writer is not None:
            if('bits' in name):
                writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
            else:
                writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
            if param.grad is not None:
                writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    
    return 

def train_conventional_fullprecision(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    
    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        
        #optimizer zero grad
        optimizer.zero_grad()
        
        
        
        # compute output and get intermediate activations of student and teacher
        output_exit_student = model(inputs)
        # print(len(output_exit_student))
        loss_ce_exit = 0
        for tt in range(0,len(output_exit_student)):
            loss = criterion(output_exit_student[tt], targets).mean()
            loss_ce_exit += loss
            if writer is not None:
                writer.add_scalar('loss_ce/exit'+str(tt), loss, base_step + i)
        
        loss_ce_exit.backward(retain_graph = True)
        
        # for name, m in model.named_parameters():
        #     if ('module.features.9.exit_conv.0.weight' in name):
        #         print(m.grad)
        
        if writer is not None:
            writer.add_scalar('loss_ce/total', loss_ce_exit, base_step + i)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_exit_student[-1], targets, topk=(1, 5))
        losses.update((loss_ce_exit).item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if writer is not None:
            if('bits' in name):
                writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
            else:
                writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
            if param.grad is not None:
                writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    
    return 



def train_quantization_loss_scaling(train_loader, teacher_model, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    
    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
            targets_onehot = targets
        if teacher_model is not None: 
            with torch.no_grad():
                targets = torch.nn.functional.softmax(teacher_model(inputs))  
        

        #optimizer zero grad
        optimizer.zero_grad()
        
        if(i==0):
            for name, m in model.named_modules():
                    if ('conv1' in name and 'layer' not in name):
                        #first layer activations to remain 8-bits and not trainable
                        m.bits_a.data.copy_(torch.ones(1) * 8)
                        m.bits_a.requires_grad = False
                        m.is_first.fill_(1)
                        break
        
        # compute output and get intermediate activations of student and teacher
        output_exit_student = model(inputs)
        # print(len(output_exit_student))

        #get loss scaling coefficients
        with torch.no_grad():
            exit_probs = []
            for tt in range(0,len(output_exit_student)-1):
                prob = torch.nn.functional.softmax(output_exit_student[tt])
                prob = prob[range(targets.shape[0]), targets]
                exit_probs.append(prob)
            # cur_coeffs = torch.stack([(1 - (0.5*(exit_probs[i-1] + exit_probs[i-2])).pow(2)) if i >=2 else torch.cuda.FloatTensor(exit_probs[0].shape).fill_(0.5) for i in range (0,5) ]).unsqueeze(2)
            cur_coeffs = torch.stack([((1+exit_probs[i]).pow(0.5)) if i < (len(output_exit_student)-2) else torch.cuda.FloatTensor(exit_probs[0].shape).fill_(1.0) for i in range (0,5)]).unsqueeze(2)

        loss_ce_exit = 0
        for tt in range(0,len(output_exit_student)):
            loss = criterion(output_exit_student[tt], targets)
            # print(cur_coeffs[tt])
            loss = (loss * cur_coeffs[tt]).mean()
            loss_ce_exit += loss
            if writer is not None:
                writer.add_scalar('loss_ce/exit'+str(tt), loss, base_step + i)
        
        loss_ce_exit.backward(retain_graph = True)
        
       
        if writer is not None:
            writer.add_scalar('loss_ce/total', loss_ce_exit, base_step + i)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_exit_student[-1], targets_onehot, topk=(1, 5))
        losses.update((loss_ce_exit).item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if writer is not None:
            if('bits' in name):
                writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
            else:
                writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
            if param.grad is not None:
                writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    
    return 





def train_ensemble_resnet18(train_loader, model, ensemble_model, teacher_model, criterion, optimizer, epoch, args, writer, target_bitops, bit_train_flag, switch, exit_id):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    # switch to train mode
    model.train()
    
    end = time.time()
    
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu,non_blocking=True)
            targets = targets.cuda(args.gpu,non_blocking=True)
        
        if teacher_model is not None:
            targets = teacher_model(inputs)
        #optimizer zero grad
        optimizer.zero_grad()
        
        # compute output and get intermediate activations of student and teacher
        output_ensemble = ensemble_model(inputs)
        #output ensemble is a list of output probabilities
        #convert to log domain for NLL loss
        loss_ce = criterion(output_ensemble.log(), targets) 
        
        loss_ce.backward()
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_ensemble, targets, topk=(1, 5))
        losses.update((loss_ce).item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            writer.add_scalar('loss/total', loss_ce, base_step + i)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        if switch:
            switch = 0
            for name, m in model.named_modules():
                if(isinstance(m, my_nn.Conv2dLSQ)):
                    m.switch.fill_(0)
        
        
    # add weight and gradient distributions to tensorboard logs       
    for name, param in model.named_parameters():
        if writer is not None:
            if('bits' in name):
                writer.add_histogram('bits/'+name,param.data,epoch* len(train_loader) + i)
            else:
                writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
            if param.grad is not None:
                writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
    
    return 
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)






def get_summary_writer(args, ngpus_per_node, model):
    if not args.hp.multi_gpu.multiprocessing_distributed or (args.hp.multi_gpu.multiprocessing_distributed
                                                             and args.hp.multi_gpu.rank % ngpus_per_node == 0):
        args.log_name = 'logger/{}_{}'.format(args.hp.arch,
                                                    args.hp.log_name)
        writer = SummaryWriter(args.log_name)
        with open('{}/{}.prototxt'.format(args.log_name, args.arch), 'w') as wf:
            wf.write(str(args.hp))
        with open('{}/{}.txt'.format(args.log_name, args.arch), 'w') as wf:
            wf.write(str(model))
        return writer
    return None


def get_model_info(model, args, input_size=(3, 224, 224)):
    print('Inference for complexity summary')
    if isinstance(input_size, torch.utils.data.DataLoader):
        input_size = input_size.dataset.__getitem__(0)[0].shape
        input_size = (input_size[0], input_size[1], input_size[2])
    # with open('{}/{}_flops.txt'.format(args.hp.log_name, args.arch), 'w') as f:
    flops, params = get_model_complexity_info(
        model, input_size, as_strings=True, print_per_layer_stat=True, ost=None)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # with open('{}/{}.txt'.format(args.log_name, args.arch), 'w') as wf:
    #     wf.write(str(model))
    # with open('{}/{}.prototxt'.format(args.log_name, args.arch), 'w') as wf:
    #     wf.write(str(args.hp))
    # summary(model, input_size)
    if args.hp.export_onnx:
        dummy_input = torch.randn(1, input_size[0], input_size[1], input_size[2], requires_grad=True).cuda(args.gpu)
        # torch_out = model(dummy_input)
        torch.onnx.export(model,  # model being run
                          dummy_input,  # model input (or a tuple for multiple inputs)
                          # where to save the model (can be a file or file-like object)
                          "{}/{}.onnx".format(args.log_name, args.arch),
                          export_params=True,  # store the trained parameter weights inside the model file
                          # opset_version=10,  # the ONNX version to export the model to
                          input_names=['input'],  # the model's input names
                          output_names=['output']  # the model's output names
                          )
    return flops, params


def save_checkpoint(state, is_best, prefix, epoch, filename='_checkpoint.pth.tar'):
    torch.save(state, prefix + "_epoch" + str(epoch) + filename)
    if is_best:
        shutil.copyfile(prefix + "_epoch" + str(epoch) + filename, prefix + 'best.pth.tar')
    return


def process_model(model, args, replace_map=None, replace_first_layer=False, **kwargs_module):
    if not hasattr(args, 'arch'):
        args.arch = args.hp.arch

    if args.hp.HasField('weight'):
        if os.path.isfile(args.hp.weight):
            print("=> loading weight '{}'".format(args.hp.weight))
            weight = torch.load(args.hp.weight, map_location='cpu')
            model.load_state_dict(weight)
        else:
            print("=> no weight found at '{}'".format(args.hp.weight))

    if replace_map is not None:
        tool = wrapper.ReplaceModuleTool(model, replace_map, replace_first_layer, **kwargs_module)
        tool.replace()
        args.replace = [tool.convs, tool.linears, tool.acts]
        print('after modules replacement')
        display_model(model)
        info = ''
        for k, v in replace_map.items():
            if isinstance(v, list):
                for vv in v:
                    info += vv.__name__
            else:
                info += v.__name__
        args.arch = '{}_{}'.format(args.arch, info)
        print('Please update optimizer after modules replacement')

    if args.hp.HasField('resume'):
        if os.path.isfile(args.hp.resume):
            print("=> loading checkpoint '{}'".format(args.hp.resume))
            checkpoint = torch.load(args.hp.resume, map_location='cpu')
            new_checkpoint = copy.deepcopy(checkpoint)

            # for key in new_checkpoint.keys():
            #     if key in checkpoint.keys():
            #         temp_ = checkpoint.pop(key)
            #         checkpoint[key.replace('module.', '')] = temp_

            for key in new_checkpoint['state_dict'].keys():
                if key in checkpoint['state_dict'].keys():
                    temp_ = checkpoint['state_dict'].pop(key)
                    checkpoint['state_dict'][key.replace('module.', '')] = temp_
            model.load_state_dict(checkpoint['state_dict'], strict = False)
        else:
            print("=> no checkpoint found at '{}'".format(args.hp.resume))

    return


class DataloaderFactory(object):
    # MNIST
    mnist = 0
    # CIFAR10
    cifar10 = 10
    # ImageNet2012
    imagenet2012 = 40
    # TinyImagenet
    tiny_imagenet = 60

    def __init__(self, args):
        self.args = args
        self.mnist_transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.cifar10_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.cifar10_transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def product_train_val_loader(self, data_type):
        args = self.args
        noverfit = not args.hp.overfit_test
        train_loader = None
        val_loader = None
        # MNIST
        if data_type == self.mnist:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(args.hp.data, train=True, download=True,
                                           transform=self.mnist_transform),
                batch_size=args.hp.batch_size, shuffle=True and noverfit)
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(args.hp.data, train=False, transform=self.mnist_transform),
                batch_size=args.hp.batch_size, shuffle=False)
            return train_loader, val_loader
        # CIFAR10
        if data_type == self.cifar10:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            trainset = torchvision.datasets.CIFAR10(root=args.hp.data, train=True, download=True,
                                                    transform=self.cifar10_transform_train)
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            else:
                train_sampler = None
            
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.hp.batch_size,
                                                       shuffle=(train_sampler is None) and noverfit,
                                                       num_workers=args.hp.workers, sampler=train_sampler, worker_init_fn=seed_worker)
            testset = torchvision.datasets.CIFAR10(root=args.hp.data, train=False, download=True,
                                                   transform=self.cifar10_transform_val)
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args.hp.batch_size, shuffle=False,
                                                     num_workers=args.hp.workers, worker_init_fn=seed_worker)
            return train_loader, val_loader, train_sampler
        # ImageNet
        elif data_type == self.imagenet2012:
            # Data loading code
            traindir = os.path.join(args.hp.data, 'train')
            valdir = os.path.join(args.hp.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.hp.batch_size, shuffle=(train_sampler is None) and noverfit,
                num_workers=args.hp.workers, pin_memory=True, sampler=train_sampler)

            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.hp.batch_size, shuffle=False,
                num_workers=args.hp.workers, pin_memory=True)
            return train_loader, val_loader, train_sampler
        elif data_type == self.tiny_imagenet:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            trainset = torchvision.datasets.CIFAR100(root=args.hp.data, train=True, download=True,
                                                    transform=self.cifar10_transform_train)
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            else:
                train_sampler = None
            
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.hp.batch_size,
                                                       shuffle=(train_sampler is None) and noverfit,
                                                       num_workers=args.hp.workers, sampler=train_sampler)
            testset = torchvision.datasets.CIFAR100(root=args.hp.data, train=False, download=True,
                                                   transform=self.cifar10_transform_val)
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args.hp.batch_size, shuffle=False,
                                                     num_workers=args.hp.workers)
            return train_loader, val_loader, train_sampler
        else:
            assert NotImplementedError
        return train_loader, val_loader


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
        self.fmt = fmt

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def distributed_model(model, ngpus_per_node, args):
    if not torch.cuda.is_available() or args.gpu is None:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(int(args.gpu))
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.hp.batch_size = int(args.hp.batch_size / ngpus_per_node)
            args.hp.workers = int(
                (args.hp.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            assert NotImplementedError
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(int(args.gpu))
        model = model.cuda(args.gpu)
    else:
        assert NotImplementedError
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.hp.arch.startswith('alexnet') or args.hp.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model

def distributed_model_teacher(model, ngpus_per_node, args):
    if not torch.cuda.is_available() or args.gpu is None:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(int(args.gpu))
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            assert NotImplementedError
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(int(args.gpu))
        model = model.cuda(args.gpu)
    else:
        assert NotImplementedError
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.hp.arch.startswith('alexnet') or args.hp.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model


def get_hash_code(message):
    hash = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash[:8]


def get_current_time():
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%Y-%m-%d-%H:%M")


def display_model(model):
    str_list = str(model).split('\n')
    if len(str_list) < 30:
        print(model)
        return
    begin = 10
    end = 5
    middle = len(str_list) - begin - end
    abbr_middle = ['...', '{} lines'.format(middle), '...']
    abbr_str = '\n'.join(str_list[:10] + abbr_middle + str_list[-5:])
    print(abbr_str)


def def_module_name(model):
    for module_name, module in model.named_modules():
        module.__name__ = module_name
