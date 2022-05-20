import torch
import torch.nn as nn

from MobileNet import *
from MyDataSet import *

from torch.utils.data import DataLoader

if __name__ == "__main__":
    print("Model load")
    model = MobileNetV1(ch_in=1, n_classes=2)
    print("Model load Complete")

    print("Model weight load")

    model.load_state_dict(torch.load('./ptfiles/20220519_149.pt'))
    print("Model weight load Complete")

    print("loading test_data")
    test_data = MyDataSet_TEST()
    print("loading test_data complete")

    test_loader = DataLoader(test_data, batch_size = 64, shuffle=False)

    model.eval()
    
    correct = 0

    print("Testing.....")
    for idx, (input, target) in enumerate(test_loader):
        input = np.array(input)
        input = torch.tensor(input, dtype=torch.float32)

        input = input.unsqueeze(1)
        input = input.float()
        input = input.cuda()
        
        output = model(input)

        tmp_correct= 0
        for i in range(len(output)):
            if(output[i] == target[i]):
                correct += 1
                tmp_correct += 1
        
        print("Current acc => ", tmp_correct / len(output))
    
    print("==========================================")
    print("Total Acc =>", correct/len(test_data))
    print("==========================================")
