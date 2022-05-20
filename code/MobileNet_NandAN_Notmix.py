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

from MobileNet import *
from MyDataSet import *

#############################################################################
################################ TRAIN CODE #################################
def main():
    print("load data.... ")
    # train_df = pd.read_csv("../UNSW_NB15_training-set.csv", index_col = False).drop('id', axis = 1)
    # test_df = pd.read_csv("../UNSW_NB15_testing-set.csv", index_col = False).drop('id', axis = 1)
    print("load data complete!")

    print("Making Dataset.... ")
    train_data = MyDataSet()
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
        if(epoch % 10 == 9):
            
            torch.save(model.state_dict(), f'./ptfiles/20220519_{epoch}.pt')
    
    
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
        # print("input length : ", len(input))

        input = np.array(input)
        target = np.array(target)
                
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target)

        input = input.unsqueeze(1)
                
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

