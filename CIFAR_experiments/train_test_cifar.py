'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models.models import *
from torch.utils.data.sampler import SubsetRandomSampler
from utils import progress_bar
import numpy as np
from cifar10_input import *

# Main training loop
## Train an epoch
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0.0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outs = net(inputs)
        loss = criterion(outs, targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _, predicted = outs.max(1)
            correct += predicted.eq(targets).sum().item()
            train_loss += loss.item()
            total += targets.size(0)
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def valid(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0.0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = net(inputs)
            loss = criterion(outs, targets)
            _, predicted = outs.max(1)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()            
            total += targets.size(0)            

        progress_bar(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss':test_loss,
        }
        torch.save(state, chckpt)
        if acc> best_acc:
            best_acc = acc
    return test_loss

def test():
    '''Compute standard accuracy'''
    checkpoint = torch.load(chckpt)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Print test accuracy.
    acc = 100.*correct/total
    print('Test accuracy', acc)
    return acc

def make_grid():
    '''returns affine grid corresponding to rotations in steps of 1 degree'''
    torch.pi = torch.acos(torch.zeros(1)).item() * 2
    rad = 2*torch.pi*torch.arange(0.0,360,1.0)/360
    cose = torch.cos(rad).unsqueeze(0)
    sine = torch.sin(rad).unsqueeze(0)
    zer0s = torch.zeros_like(cose)
    rotmats = torch.cat([torch.cat([cose,-sine,zer0s],0).unsqueeze(0),
                     torch.cat([sine,cose,zer0s],0).unsqueeze(0)],0).permute(2,0,1).to(device)
    grid = F.affine_grid(rotmats, (360,3,32,32)) 
    return grid

def adversarial_test(net,interp_mode,grid): 
    net.eval()
    counts = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs=inputs.repeat(360,1,1,1)  
                      
            #rotate input image in steps of 1 degree
            inputs = F.grid_sample(inputs, grid, mode=interp_mode, padding_mode='zeros')

            outs = net(inputs)
            _, preds = outs.max(1)
            #Number of instances with atleast one error among 360 rotations
            counts += torch.sum(torch.sum(preds!=targets)>0).item()
            correct += torch.sum(preds==targets).item()
        #Average accuracy    
        rot_acc = np.true_divide(correct,360*len(testset))*100. 
        #Worst case accuracy
        counts = np.true_divide(counts,len(testset))*100.#Worst case err      
        adv_acc = 100.-counts #worst-case accuracy
    return rot_acc,adv_acc
    
def GA(net,gr):
    '''Use a gradient based orbit mapping prior as the first layer'''
    net1 = nn.Sequential(gr,net)
    net1 = net1.to(device)
    return net1 
   
#####Options
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset',default='CIFAR10')
parser.add_argument('--augment', default='standard', help='data augmentation-- standard, rot-default, combi-rot')
parser.add_argument('--train_batch_size',default=128,type=int)
parser.add_argument('--val_batch_size',default=100,type=int)
parser.add_argument('--gradalign',action='store_true',help='Gradient based orbit mapping module')
parser.add_argument('--net', default='Resnet18',help='network-options--linear,convnet,Resnet18')
parser.add_argument('--chkpt_dir', default = './checkpoint/',  help='Check point directory')
parser.add_argument('--run',type=int, default = None,)
parser.add_argument('--eval_mode',default=None,help='interpolation mode to evaluate rotations,options-- bilinear,nearest,bicubic')#default None evaluates all modes

args = parser.parse_args()
print(args)
chckpt = f'{args.chkpt_dir}{args.dataset}.{args.net}.{args.augment}.GA-{args.gradalign}.run-{args.run}.pth'
print(chckpt)
device = 'cuda:{}'.format(0)
best_acc = 0  # best val accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
end_epoch = 150

# Data loaders
print('==> Preparing data..')
trainloader,validloader = make_dataset(args)

# Model
print('==> Building model..')
net = get_model(args.net)
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True


#Alignment module for orbit mapping
if args.gradalign:
    gr = StableGradRot(False,[32,32],'bicubic')
    net = GA(net,gr)
    net = net.to(device)
    
#Resume from checkpoint
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(chckpt)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,verbose=True)

###Train loop###
for epoch in range(start_epoch, end_epoch):
    train(epoch)
    val_loss=valid(epoch)
    scheduler.step(val_loss)


#####Evaluate trained network###
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

#standard accuracy
print('test_normal')
std_acc = test()

fname = f'{args.net}_{args.augment}_{args.gradalign}_{args.run}.txt'
f=open(fname, "w+")
f.write(f'{chckpt}\n')
f.write(f'Std_acc:\t{std_acc}\n')

####Evaluation on Grid
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
grid = make_grid()
checkpoint = torch.load(chckpt)
net.load_state_dict(checkpoint['net'])
net.eval() 
if args.eval_mode is not None:
    interps = [args.eval_mode]
else:
    interps = ['bilinear','nearest','bicubic'] 
    
for mode in interps:
    rot_acc,adv_acc = adversarial_test(net,mode,grid)
    f.write(f'\n{mode}:\t{rot_acc}\t{adv_acc}')
f.close()

