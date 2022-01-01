#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
import torch
from torchsummary import summary
from nets.yolact import Yolact

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = Yolact(81, train_mode=True).to(device)
    summary(m, input_size=(3, 544, 544))
