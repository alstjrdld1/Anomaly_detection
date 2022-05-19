import time
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

import cv2

import torch
import torch.nn as nn 

import torchvision 
import torchvision.transforms as transforms 

from torch.utils.data import Dataset, DataLoader

from PIL import Image

#############################################################################
################################## UTILS ####################################
class AverageMeter(object):
    r"""Computes and stores the average and current value
    """
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

def accuracy(output, target, topk=(1,)):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, idx = output.sort(descending=True)
        
        pred = idx[:,:maxk]
        
        pred = pred.t()
        correct = pred.eq(target.t())

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def just_convert_to_bin(number):
    if type(number) == str: # 문자열의 경우 
        binary_value = ""
        
        for char in number :
            binary_value += bin(ord(char)).lstrip("0b")
            
        binary_value = binary_value
        return binary_value
    
    elif type(number) == int: # 정수형의 경우 그냥 binary로 바꾸고 '0b'제거 
        return bin(number).lstrip("0b")
    
    else:
        float_length = 64
        formatted_number = "{:.64f}".format(number)
        
        # 필요한 만큼 타입 변환 
        dec, float_number = str(formatted_number).split(".")
        
        # 정수부는 이진수로 바로 바꿈 
        dec = int(dec)
        res = bin(dec).lstrip("0b")
        
        # 소수부 연산 처리 
        while(len(res) < float_length):
            float_number = float("0." + float_number)            
            float_number = float_number * 2
            float_number = "{:.64f}".format(float_number)
            dec, float_number = str(float_number).split(".")
            res += dec
        return res

def make_patch(item, patch_size):
    '''
    item should be a np.ndarray 
    '''
    patch = ""
    total_length = patch_size[0] * patch_size[1]

    for elem in item : 
        patch += just_convert_to_bin(elem)

    while(len(patch) < total_length) : # patch사이즈를 일정하게 만드는 거 
        patch+= "0"

    patch = list(map(int, patch))
    patch = np.array(patch)[:total_length] # 만약 바이너리로 변형한 부분이 packet 사이즈 보다 크면 뒤는 버려버리는 것 

    return patch.reshape(patch_size)

class PacketFeature:
    def __init__(self, feature_size):
        self.frame = np.zeros(feature_size)
        self.fsize = feature_size
        # print("Frame shape : ", self.frame.shape)
        # print("Frame size : ", self.fsize)
        self.patch_count = 0

    def append(self, patch):
        size = patch.shape # Ex 32 * 32
        stride = size[0]
        try:
            if ((self.fsize[0] % stride) == 0):
                pass
            else : 
                raise
        except:
            print("frame size and patch size unmatched")
            return
        
        if(self.patch_count >= stride*stride):
            self.patch_count = 0
            
        count = self.fsize[0] // stride
        row = self.patch_count//count  # 만약 self.patch_count = 3 이면 patch row는 0~31에 내용이 들어가야하고 col에는 96~127에 있어야지 
        col = self.patch_count % count

        for row_stride in range(stride):
            current_row = row*stride + row_stride
            current_col_start = col*stride
            current_col_end = current_col_start + stride

            self.frame[current_row][current_col_start:current_col_end] = patch[row_stride]
        
        self.patch_count = self.patch_count + 1
#############################################################################
#############################################################################


#############################################################################
################################## Model ####################################
class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
#############################################################################
#############################################################################


#############################################################################
######################### Dataset Preprocessing #############################
class MyDataSet(Dataset):
    def __init__(self, df):
        packets = df.drop(['attack_cat', 'label'], axis=1).values
        self.x_train = []
        self.y_train = []
        y_train = df.iloc[:, [-1]].values
        
        pf = PacketFeature((224, 224))
        for idx, _ in enumerate(packets):
            if(idx+49 > len(packets)):
                break

            sum = 0
            for count in range(49):
                patch = make_patch(packets[idx + count], (32, 32))
                pf.append(patch)
                sum += int(y_train[idx+count])

            if(sum != 0):
                self.y_train.append(1)
            else:
                self.y_train.append(0)

            self.x_train.append(pf.frame)
  
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
#############################################################################
#############################################################################


#############################################################################
################################ TRAIN CODE #################################
def main():
    print("load data.... ")
    train_df = pd.read_csv("../UNSW_NB15_training-set.csv", index_col = False).drop('id', axis = 1)
    # test_df = pd.read_csv("../UNSW_NB15_testing-set.csv", index_col = False).drop('id', axis = 1)
    print("load data complete!")

    print("Making Dataset.... ")
    train_data = MyDataSet(train_df)
    print("Making Dataset complete! ")

    WEIGHTDECAY = 1e-4
    MOMENTUM = 0.9
    BATCHSIZE = 64
    LR = 0.1
    EPOCHS = 150

    train_loader = DataLoader(train_data, batch_size = BATCHSIZE, shuffle=True)
    model = MobileNetV1(ch_in=1, n_classes=2)

    optimizer = torch.optim.SGD(model.parameters(), lr = LR,
                               momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                               nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [75,125], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    model = model.cuda()
    criterion = criterion.cuda()
    
    # parameter of our model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    
    last_top1_acc = 0
    
    for epoch in range(EPOCHS):
        print("\n----- epoch: {}, lr: {} -----".format(
        epoch, optimizer.param_groups[0]["lr"]))
        
        # train for one epoch 
        start_time = time.time()
#         last_top1_acc = train(train_loader, epoch, model, optimizer, criterion)
        train(train_loader, epoch, model, optimizer, criterion)
        elapsed_time = time.time() - start_time 
        print('==> {:.2f} seconds to  train this epoch \n'.format(
                elapsed_time))
        
        # learning rate scheduling 
        scheduler.step()
        if(epoch % 10 == 0):
            
            torch.save(model.state_dict(), f'/ptfiles/20220519_{epoch}.pt')
    
    
#     print(f"Last Top-1 Accuracy: {last_top1_acc}")
#     print(f"Number of parameters: {pytorch_total_params}")
    
def train(train_loader, epoch, model, optimizer, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time,  data_time, losses, 
                             top1, prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode 
    model.train()
    end = time.time()
    
    batch_loss = []
    total = 0 
    correct = 0
    best_acc = 0
    PRINTFREQ = 20
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time 
        data_time.update(time.time() - end)
        print("input length : ", len(input))

        input = np.array(input)
        target = np.array(target)
                
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
                
        input = input.float()
        input = input.cuda()
        target = target.cuda()
        target = target.squeeze(-1)
        
        # compute ouput 
        output = model(input)

        loss = criterion(output, target)
        _, predicted = output.max(1)
        total += target.size(0)
        
        correct += predicted.eq(target).sum().item()
        acc1 = correct/total

        # measure accuracy and record loss, accuracy         
        losses.update(loss.item(), input.size(0))
        
        # compute gradient and do SGD step 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time 
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % PRINTFREQ == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, i * len(input), len(train_loader.dataset, ),
                       100. * i / len(train_loader), loss.item(), 100. * correct / total))
        batch_loss.append(loss.item())
        loss_avg = sum(batch_loss) / len(batch_loss)
#############################################################################
#############################################################################

if __name__ == "__main__":
    main()

