from __future__ import print_function
import os
import argparse
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as sched
from util.array_util import mean_dim
from util.optim_util import bits_per_dim, clip_grad_norm, NLLLoss
from util.shell_util import AverageMeter
import utils
from utils import *
from tqdm import tqdm
import utils.optim_util as op
from models.glow.glow import Glow
from utils.dataset import  glow_dataset
#from glow_dataset import  glow_dataset
import pickle


##### train, test, retrain function #######################################################
# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             #test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
#             test_loss += F.cross_entropy(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#     prec = 100. * correct / len(test_loader.dataset)
#     print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset), prec))
#
#     return prec
def test(model, device, testloader, loss_fn, num_class=0):
    global best_loss
    model.eval()
    loss_meter = AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, _ in testloader:
            x = x.to(device)
            z, sldj = model(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': model.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/' + str(num_class) + '.pth.tar'.format(epoch))
        best_loss = loss_meter.avg

    # Save samples and data
    # images = sample(dodel, num_samples, device)
    os.makedirs('samples', exist_ok=True)
    os.makedirs('samples/' + str(num_class), exist_ok=True)
    return best_loss
    # images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    # torchvision.utils.save_image(images_concat, 'samples/'+str(num_class)+'/epoch_{}.png'.format(epoch))


# def retrain(args, model, mask, device, train_loader, test_loader, optimizer):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         # loss = regularized_nll_loss(args, model, F.log_softmax(output, dim=1), target)
#         loss = regularized_nll_loss(args, model, output, target)
#         loss.backward()
#         optimizer.prune_step(mask)

def retrain(args, model, mask,trainloader, device, optimizer, glow_optimizer,loss_fn):
    global_step=0
    print('\nEpoch: %d' % epoch)
    model.train()
    loss_meter = AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            glow_optimizer.zero_grad()
            z, sldj = model(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if args.max_grad_norm > 0:
                clip_grad_norm(glow_optimizer, args.max_grad_norm)
            glow_optimizer.step()
            scheduler.step(global_step)
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=bits_per_dim(x, loss_meter.avg),
                                     lr=glow_optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)
    optimizer.prune_step(mask)

if __name__ == '__main__':
    ##### Settings #########################################################################
    parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
    parser.add_argument('--model', default='glow', help='select model')
    parser.add_argument('--dir', default='~/data', help='dataset root')
    parser.add_argument('--dataset', default='glow_cifar10', help='select dataset')
    parser.add_argument('--batchsize', default=16, type=int, help='set batch size')
    parser.add_argument('--lr', default=3e-4, type=float, help='set learning rate')
    parser.add_argument('--re_lr', default=1e-4, type=float, help='set fine learning rate')
    parser.add_argument('--alpha', default=5e-4, type=float, help='set l2 regularization alpha')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='adam epsilon')
    parser.add_argument('--rho', default=1e-2, type=float, help='set rho')
    parser.add_argument('--connect_perc', default=3.6, type=float, help='connectivity pruning ratio')
    parser.add_argument('--epoch', default=2, type=int, help='set epochs')
    parser.add_argument('--re_epoch', default=2, type=int, help='set retrain epochs')
    parser.add_argument('--num_sets', default='8', type=int, help='# of pattern sets')
    parser.add_argument('--exp', default='test', type=str, help='test or not')
    parser.add_argument('--l2', default=False, action='store_true', help='apply l3 regularization')
    parser.add_argument('--scratch', default=False, action='store_true', help='start from pretrain/scratch')
    parser.add_argument('--no-cuda', default=False, action='store_true', help='disables CUDA training')
    ###
    parser.add_argument('--warm_up', default=400000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--glow_lr', default=1e-3, type=float, help='glow_Learning rate')
    args = parser.parse_args()
    print(args)
    comment = "check7"

    if args.exp == 'test':
        args.exp = f'{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
    args.save = f'logs/{args.dataset}/{args.model}/{args.exp}_lr{str(args.lr)}_rls{str(args.re_lr)}_{comment}'
    print(args.save)

    args.workers = 16

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ##########################################################################################################

    print('Preparing pre-trained model...')
    if args.dataset == 'imagenet':
        pre_model = torchvision.models.vgg16(pretrained=True)
    elif args.dataset == 'cifar10':
        pre_model = utils.__dict__[args.model]()
        # pre_model.load_state_dict(torch.load('./cifar10_pretrain/vgg16_bn.pt'), strict=True)
    elif args.dataset == 'glow_cifar10':
        pre_model = Glow(num_channels=512,
               num_levels=4,
               num_steps=8)
    print("pre-trained model:\n", pre_model)

    ##### Find Pattern Set #####
    print('\nFinding Pattern Set...')
    # 改成导入glow的

    if os.path.isfile('pattern_set_'+ args.model + '.npy') is False:
        pattern_set = pattern_setter(pre_model)
        np.save('pattern_set_'+ args.dataset + '.npy', pattern_set)
    else:
        pattern_set = np.load('pattern_set_'+ args.model+ '.npy')
    # pattern_set = pattern_setter(pre_model)
    # np.save('pattern_set_' + args.model + '.npy', pattern_set)
    #pattern_set = np.load('l4_16.pth.tar')
    pattern_set = pattern_set[:args.num_sets, :]
    print(pattern_set)
    print('pattern_set loaded')

    ##### Load Dataset ####

    print('\nPreparing Dataset...')
    train_loader, test_loader = data_loader(args.dir, args.dataset, args.model, args.batchsize, args.workers)

    print('Dataset Loaded')

    ##### Load Model #####
    #model = utils.__dict__[args.model]()
    model =pre_model
    # if pre-trained... load pre-trained weight
    if not args.scratch:
        state_dict = pre_model.state_dict()
        torch.save(state_dict, 'tmp_pretrained.pt')

        model.load_state_dict(torch.load('tmp_pretrained.pt'), strict=True)
    model.cuda()
    pre_model.cuda()

    # History collector
    history_score = np.zeros((args.epoch + args.re_epoch + 1, 2))

    print('lr:', args.lr, 'rho:', args.rho)
    print('\nBefore pruning test...')
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ####
    best_loss =1e5
    loss_fn = op.NLLLoss().to(device)
    #test(model, device, test_loader, loss_fn, 0)

    create_exp_dir(args.save)
    torch.save(model.state_dict(), os.path.join(args.save, 'glow_before.pth.tar'))

    # Real Pruning ! ! !
    print("\nApply Pruning with connectivity & pattern set...")
    mask = apply_prune(args, model, device, pattern_set)
    print_prune(model)

    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4:
            param.data.mul_(mask[name])

    torch.save(model.state_dict(), os.path.join(args.save, 'glow_cifar10_after.pth.tar'))

    print("\nAfter pruning test...")

    #test(model, device, test_loader, loss_fn, 0)

    # Optimizer for Retrain
    optimizer = PruneAdam(model.named_parameters(), lr=args.re_lr, eps=args.adam_epsilon)
    glow_optimizer = optim.Adam(model.parameters(), lr=args.glow_lr)
    scheduler = sched.LambdaLR(glow_optimizer, lambda s: min(1., s / args.warm_up))
    # Fine-tuning...
    print("\nfine-tuning...")
    best_prec = 0
    for epoch in range(args.re_epoch):
        # print("Epoch: {} with re_lr: {}".format(epoch+1, args.re_lr))
        if epoch in [args.re_epoch // 4, args.re_epoch // 2, args.re_epoch // 4 * 3]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        #retrain(args, model, mask, device, train_loader, test_loader, optimizer)
        retrain(args, model, mask,train_loader, device, optimizer, glow_optimizer,loss_fn)
        # print("\ntesting...")
        prec = test(model, device, test_loader, loss_fn, 0)
        history_score[args.epoch + epoch][0] = epoch
        history_score[args.epoch + epoch][1] = prec

        if prec < best_prec:
            best_prec = prec
            torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_pruned.pth.tar'))

    np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')

############################################

# my mistake 1 - making mask.pickle
"""
with open('mask.pickle', 'wb') as fw:
    pickle.dump(mask, fw)

with open('mask.pickle', 'rb') as fr:
    mask = pickle.load(fr)
    print("mask loaded")
"""

# my mistake 2
"""
for module in model.named_modules():
    if isinstance(module[1], nn.Conv2d):
        print("module:", module[0])
        prune.custom_from_mask(module, 'weight', mask=mask[module[0] +'.weight'])
"""
