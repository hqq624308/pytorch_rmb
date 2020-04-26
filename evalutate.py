# evaluate on GPU
# Usage: python evalutate.py  -a shufflenet_v2_x1_0
import os  
import torch.nn as nn 
import torchvision.models as models
import torch
import argparse
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class shuffle_my(nn.Module):
    def __init__(self):
        super(shuffle_my,self).__init__()
        model = models.__dict__[args.arch](pretrained=False)
        self.new = nn.Sequential(*list(model.children())[:-1])
        self.pool_layer = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(1024,2)

    def forward(self,x):
        x = self.new(x)
        x = self.pool_layer(x)
        x = x.view(x.size(0),-1)
        x_feat = x
        x = self.fc(x)

        return x_feat,x

class my_dataset(Dataset):
    def __init__(self,path,transform=None):
        contents = open(path,'r')
        imgs = []
        for line in contents:
            line = line.strip('\n')
            imgs.append(line)
        self.img = imgs
        self.transform = transform
    
    def __getitem__(self,index):
        img_path = self.img[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,img_path
    
    def __len__(self):
        return len(self.img)

def eval_hu(model,val_loader):
    model.eval()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) 

    result = './result.txt'
    f = open(result,'w')
    path_rec = []
    score_rec = []
    for _,(data,path_tem) in tqdm(enumerate(val_loader)):
        data = data.cuda()
        with torch.set_grad_enabled(False):
            #print(data.shape)
            _,output = model(data)
            #print(output.shape)
            score = F.softmax(output,dim=1)
            #print(score[:,1])
            path_rec.extend(path_tem)
            score_rec.extend(score[:,1].cpu().numpy())
    
    for i in range(len(score_rec)):
        if score_rec[i]>0.5:
            f.write(path_rec[i] + ' ' + str(score_rec[i]) + '\n')

    f.close()

if __name__ == '__main__':
    #model_path = './model_best.pth.tar'
    model_path = './checkpoint.pth.tar'
    testset = './test3008.txt'

    val_loader = torch.utils.data.DataLoader(
            my_dataset(testset, transforms.Compose([
                transforms.Resize(size=(224,224),interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                [0.229,0.224,0.225]),
            ])),batch_size=8, shuffle=False,num_workers=4, pin_memory=True)

    #define and load the model
    model = shuffle_my()
    best_model = torch.load(model_path,map_location=lambda storage, loc: storage)
    state_dict = best_model["state_dict"]
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    model=model.cuda()
    eval_hu(model,val_loader)
