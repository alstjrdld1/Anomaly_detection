import torch 
import numpy as np 
import pandas as pd

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