import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import OrderedDict
from .Gaussian import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def get_model(model_name):
    """Retrieve an appropriate architecture."""
    if model_name=='Resnet18':
        model = ResNet18()
    elif model_name == 'convnet':
        model = convnet(width=32, in_channels=3, num_classes=10)
    elif model_name ==  'linear':
        model = linear_model('CIFAR10', num_classes=10)
    return model



###Linear model    
def linear_model(dataset, num_classes=10):
    """Define the simplest linear model."""
    if 'cifar' in dataset.lower():
        dimension = 3072
    elif 'mnist' in dataset.lower():
        dimension = 784
    elif 'imagenet' in dataset.lower():
        dimension = 150528
    elif 'tinyimagenet' in dataset.lower():
        dimension = 64**2 * 3
    else:
        raise ValueError('Linear model not defined for dataset.')
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(dimension, num_classes))

####Simple Convnet model
def convnet(width=32, in_channels=3, num_classes=10, **kwargs):
    """Define a simple ConvNet. This architecture only really works for CIFAR10."""
    model = torch.nn.Sequential(OrderedDict([
        ('conv0', torch.nn.Conv2d(in_channels, 1 * width, kernel_size=3, padding=1)),
        ('relu0', torch.nn.ReLU()),
        ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu1', torch.nn.ReLU()),
        ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu2', torch.nn.ReLU()),
        ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu3', torch.nn.ReLU()),
        ('pool3', torch.nn.MaxPool2d(3)),
        ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu4', torch.nn.ReLU()),
        ('pool4', torch.nn.MaxPool2d(3)),
        ('flatten', torch.nn.Flatten()),
        ('linear', torch.nn.Linear(36 * width, num_classes))
    ]))
    return model
    
    
#####ResNet Blocks####    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class FastbatchcolorimageInterpNet(nn.Module):
    def __init__(self,img,grad_required=True):
        super(FastbatchcolorimageInterpNet, self).__init__()
        self.imgs = img
        self.device = 'cuda'#self.imgs.device
        self.b,self.nc,self.ny,self.nx = img.shape
        self.rotMatrixbatch = torch.eye(2).repeat(self.b,1,1).to(self.device)
        self.grad_required = grad_required
        
        
    def forward(self, z):
        ## needs a better model than bilinear interpolation ...        

        num_pts = z.shape[0]
        
        x0 = z.clone().detach().to(self.device) #batch of points# 
            
        x0[:,0] = x0[:,0]*(self.ny-1)
        x0[:,1] = x0[:,1]*(self.nx-1)

        ###Handling points out of image boundaries 
        ###coordinates out of boundary set to 0.0, results and grads at locs will be reset to 0.0 subsequently
        locs = torch.logical_or(torch.logical_or(x0[:,0]<0.0,x0[:,0]>self.ny-1),
                              torch.logical_or(x0[:,1]<0.0,x0[:,1]>self.nx-1))
        
        x0[locs]=0.0
        
        #floor--grid points
        y_grid = torch.floor(x0[:,0]).clone()
        y_grid = y_grid.long()
        x_grid = torch.floor(x0[:,1]).clone()
        x_grid = x_grid.long()
            
        
        
        if self.grad_required:  
            grad_calc =torch.zeros(self.b,self.nc, 2, num_pts).to(self.device)
            a1 = self.imgs[:,:,y_grid+1,x_grid]-self.imgs[:,:,y_grid,x_grid]
            a2 = self.imgs[:,:,y_grid+1,x_grid+1]-self.imgs[:,:,y_grid,x_grid+1]
            a3 = self.imgs[:,:,y_grid,x_grid+1]-self.imgs[:,:,y_grid,x_grid]
            
            grad_calc = torch.cat([((a1-a2)*(x_grid-x0[:,1])).unsqueeze(2),
                           ((a1-a2)*(y_grid-x0[:,0])).unsqueeze(2)],2)+torch.cat([a1.unsqueeze(2),a3.unsqueeze(2)],2)
            
            #Handling out of boundary   
            grad_calc[:,:,:,locs] = 0.0 
            
            return  torch.sum(grad_calc,(1,3))

        else:
            temp1 = (x0[:,0] - y_grid)*self.imgs[:,:,y_grid+1,x_grid] + (1 - x0[:,0] +y_grid)*self.imgs[:,:,y_grid,x_grid]
            temp2 = (x0[:,0] - y_grid)*self.imgs[:,:,y_grid+1,x_grid+1] + (1 - x0[:,0]+y_grid)*self.imgs[:,:,y_grid,x_grid+1]
            res = (x0[:,1] - x_grid)*temp2 + (1 - x0[:,1] + x_grid)*temp1
            #Handling out of boundary for image interpolation
            res[:,:,locs] = 0.0
            return res
    
       
    def applyMatrix(self,mats):
        self.rotMatrixbatch = torch.matmul(mats,self.rotMatrixbatch)



        
        
class StableGradRot(nn.Module):
    """Module which rotates the input images based on gradient at random points.
    """    
    def __init__(self,mask=False,inshape=[128,128],mode = 'bilinear',):
        super(StableGradRot, self).__init__()    
              
        self.interpnet = FastbatchcolorimageInterpNet
        self.n = 300 #num points on a circle     
        self.gauss = GaussianBlur((5, 5), (1.5, 1.5))
        self.mask = mask
        self.inshape = inshape
        self.circmask = self.make_mask()
        self.interpmode = mode
                
    def rendercolorImage(self,rotatedImgFunc,ny,nx,device):
        grid = F.affine_grid(torch.cat([torch.eye(2),
                                        torch.tensor([[0.0],[0.0]])],1).repeat(1,1,1).to(device), [1, 3, ny, nx])
        vals =  rotatedImgFunc(grid.view(-1,2))
        
        rotatedImg = vals.view(rotatedImgFunc.imgs.shape).detach()
        return rotatedImg
    
    def rotate(self,blurred_inputs,inputs):      
        
        b,nc,ny,nx = inputs.shape
        device = inputs.device
        results = torch.zeros_like(inputs)
        
        imgFunc = self.interpnet(blurred_inputs.detach(),grad_required=True) 
            
        r1 = 0.05
        r2 = 0.4
        rad = (2*torch.pi*torch.arange(0.0,self.n)/self.n).to(device)
        a = torch.cos(rad).unsqueeze(0)
        b = torch.sin(rad).unsqueeze(0)
        
        rotMats = (torch.cat([torch.cat([a,-b],0).unsqueeze(0), 
                             torch.cat([b,a],0).unsqueeze(0)],0).permute(2,0,1)).to(device)
             
        coords = (torch.cat([rotMats@torch.tensor([r1,r1]).to(device),
                             rotMats@torch.tensor([r2,r2]).to(device)],0))+ torch.tensor([0.5, 0.5]).to(device)
        
        
        avg_grad = imgFunc(coords)
        avg_grad = avg_grad/torch.norm(avg_grad, dim=1).unsqueeze(0).repeat(2,1).permute(1,0).detach()
        
        sine = avg_grad[:,0].unsqueeze(0)
        cose = avg_grad[:,1].unsqueeze(0)

        
        #imgFuncSharp = self.interpnet(inputs.detach(),grad_required=False) 
        #batch_mats = torch.cat([torch.cat([cose,-sine],0).unsqueeze(0), 
        #                     torch.cat([sine,cose],0).unsqueeze(0)],0).permute(2,0,1)
        #imgFuncSharp.applyMatrix(batch_mats)
        #results = self.rendercolorImage(imgFuncSharp,ny,nx,device)

        #Interpolation with affine_grid function
        zer0s=torch.zeros_like(cose)
        batch_mats = (torch.cat([torch.cat([cose,-sine,zer0s],0).unsqueeze(0), 
                          torch.cat([sine,cose,zer0s],0).unsqueeze(0)],0).permute(2,0,1)).to(device)
        grid = F.affine_grid(batch_mats, inputs.size())
        results = F.grid_sample(inputs.detach(), grid,mode=self.interpmode, padding_mode='zeros')

            
        return results
    
    def make_mask(self):
        ny,nx = self.inshape

        center = (int(ny/2), int(nx/2))
        radius = min(center[0], center[1], ny-center[0], nx-center[1])
        Y, X = torch.meshgrid(torch.arange(0.0,nx), torch.arange(0.0,ny))
        dist_from_center = torch.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
                   
        return mask
                    
    def forward(self,image_batch):
        imgs_blurry = self.gauss(image_batch).detach()
        with torch.no_grad():
            rotated = self.rotate(imgs_blurry,image_batch)
            if self.mask:
                rotated = rotated*(self.circmask.to(image_batch.device))
        
        return rotated         
