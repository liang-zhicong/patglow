from __future__ import print_function

import argparse

import torch
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
from torchvision import datasets, transforms
import utils
from utils import *
from tqdm import tqdm
from util.optim_util import bits_per_dim, clip_grad_norm, NLLLoss
from util.shell_util import AverageMeter
import utils
import torch.optim.lr_scheduler as sched
from utils import *
from tqdm import tqdm
import utils.optim_util as op
from models.glow.glow import Glow
import torch.backends.cudnn as cudnn
import utils.optim_util as op


##### train, test, retrain function #######################################################
def train(args, model, device, pattern_set, train_loader, test_loader, optimizer, Z, Y, U, V):
    model.train()
    for x, _ in train_loader:
        #train_loader:训练数太大
        x = x.to(device)
        optimizer.zero_grad()
        z, sldj = model(x, reverse=False)
        glow_loss = loss_fn(z, sldj)
        loss = admm_loss(args, device, model, Z, Y, U, V, glow_loss)
        loss.backward()
        optimizer.step()

    X = update_X(model)

    print("update Z, Y, U, V...")
    Z = update_Z(X, U, pattern_set, args)
    print("updated Z")
    Y = update_Y(X, V, args)
    print("updated Y")

    U = update_U(U, X, Z)
    V = update_V(V, X, Y)
    print("updated U V")


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


    return loss



def retrain(args, model, mask,trainloader, device, optimizer, loss_fn):
    global_step=0
    print('\nEpoch: %d' % epoch)
    model.train()
    loss_meter = AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = model(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if args.max_grad_norm > 0:
                clip_grad_norm(optimizer, args.max_grad_norm)
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=bits_per_dim(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)
            optimizer.prune_step(mask)

if __name__ == "__main__":
    ##### Settings #########################################################################
    parser = argparse.ArgumentParser(description='Pytorch PatDNN training glow')
    parser.add_argument('--model', default='glow', help='select model')
    parser.add_argument('--dir', default='~/data', help='dataset root')
    parser.add_argument('--dataset', default='glow_cifar10', help='select dataset')
    parser.add_argument('--batchsize', default=32, type=int, help='set batch size')
    parser.add_argument('--lr', default=3e-4, type=float, help='set learning rate')
    parser.add_argument('--re_lr', default=3e-4, type=float, help='set fine learning rate')
    parser.add_argument('--alpha', default=5e-4, type=float, help='set l2 regularization alpha')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='adam epsilon')
    parser.add_argument('--rho', default=0.01, type=float, help='set rho')#惩罚参数
    parser.add_argument('--connect_perc', default=1.5, type=float, help='connectivity pruning ratio')
    parser.add_argument('--epoch', default=20, type=int, help='set epochs')
    parser.add_argument('--re_epoch', default=20, type=int, help='set retrain epochs')
    parser.add_argument('--num_sets', default='8', type=int, help='# of pattern sets')
    parser.add_argument('--exp', default='test', type=str, help='test or not')
    parser.add_argument('--l2', default=False, action='store_true', help='apply l3 regularization')
    parser.add_argument('--scratch', default=False, action='store_true', help='start from pretrain/scratch')
    parser.add_argument('--no-cuda', default=False, action='store_true', help='disables CUDA training')

    parser.add_argument('--warm_up', default=400000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    args = parser.parse_args()

    print(args)
    comment = "check6"


    if args.exp == 'test':
        args.exp = f'{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
    args.save = f'logs/{args.dataset}/{args.model}/{args.exp}_rho{str(args.rho)}_lr{str(args.lr)}_rls{str(args.re_lr)}_{comment}_epoch{str(args.epoch)}_reepoch{str(args.re_epoch)}'
    create_exp_dir(args.save)
    data = '进行re_train训练，重新训练的epoch是：'+str(args.re_epoch)
    with open(os.path.join(args.save, 'canshu.txt'), 'w', encoding='utf-8') as f:
        f.write(data)
        f.close()
    args.workers = 16

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    ##########################################################################################################

    print('Preparing pre-trained model...')
    if args.dataset == 'imagenet':
        pre_model = torchvision.models.vgg16(pretrained=True)
    elif args.dataset == 'cifar10':
        pre_model = utils.__dict__[args.model]()
        pre_model.load_state_dict(torch.load('./cifar10_pretrain/vgg16_bn.pt'), strict=True)
    elif args.dataset == 'glow_cifar10':
        pre_model = Glow(num_channels=512,
                   num_levels=4,
                   num_steps=16)
        pre_model.load_state_dict({k.replace('module.', ''): v for k, v in
                             torch.load("l4_16.pth.tar")['net'].items()})

        print("pre-trained model:\n", pre_model)
    print("pre-trained model:\n", pre_model)

    ##### Find Pattern Set #####
    print('\nFinding Pattern Set...')
    if os.path.isfile('pattern_set_' + args.model + '.npy') is True:
        pattern_set = pattern_setter(pre_model)
        np.save('pattern_set_' + args.model + '.npy', pattern_set)
    else:
        pattern_set = np.load('pattern_set_' + args.model + '.npy')

    pattern_set = pattern_set[:args.num_sets, :]
    print(pattern_set)
    print('pattern_set loaded')

    ##### Load Dataset ####
    print('\nPreparing Dataset...')
    train_loader, test_loader = data_loader(args.dir, args.dataset,args.model, args.batchsize, args.workers)
    print('Dataset Loaded')

    ##### Load Model #####
    model = Glow(num_channels=512,
                   num_levels=4,
                   num_steps=16)
    model.load_state_dict(torch.load("20glowtrain_process.pth.tar"))

    model = model.to(device)

    # History collector
    history_score = np.zeros((args.epoch + args.re_epoch + 1, 2))


    best_loss =1e10
    loss_fn = op.NLLLoss().to(device)

    # Real Pruning ! ! !
    print("\nApply Pruning with connectivity & pattern set...")
    mask = apply_prune(args, model, device, pattern_set)
    print_prune(model)

    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[-2] != "mid_conv" and len(param.shape) == 4:
            param.data.mul_(mask[name])

    torch.save(model.state_dict(), os.path.join(args.save, 'glow_after.pth.tar'))
    X = update_X(model)
    print("\ntesting...")
    test(model, device, test_loader, loss_fn, 0)

    # Optimizer for Retrain
    optimizer = PruneAdam(model.named_parameters(), lr=args.re_lr, eps=args.adam_epsilon)

    # Fine-tuning...
    print("\nfine-tuning...")
    best_loss = 1e10
    for epoch in range(args.re_epoch):
        print("Epoch: {} with re_lr: {}".format(epoch + 1, args.re_lr))

        retrain(args, model, mask,train_loader, device, optimizer,loss_fn)

        print("\ntesting...")
        loss = test(model, device, test_loader, loss_fn, 0)
        print(loss)
        history_score[args.epoch + epoch][0] = epoch
        history_score[args.epoch + epoch][1] = loss
        if (epoch % 5 == 0):
            torch.save(model.state_dict(), os.path.join(args.save, f'{str(epoch)}_glow_process.pth.tar'))

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(args.save, 'glow_pruned.pth.tar'))

    np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')
    print("这是对l4_16的模型进行训练,剪枝，再训练后的模型，retrain_epoch是10")

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
