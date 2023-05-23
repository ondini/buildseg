from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

import sys
sys.path.append("/home/kafkaon1/FVAPP/third_party")
from projectRegularization import GeneratorResNet,Encoder, regularization

def createDeepLabv3(outputchannels):
    """ DeepLabv3
    # Args
        outputchannels: number of classes
    # Rets:
        The DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)

    model.train()
    return model


def createDeepLabv3Plus(outputchannels):
    """ DeepLabv3Plus
    # Args
        outputchannels: number of classes
    # Rets:
        The DeepLabv3Plus model with the ResNet101 backbone.
    """
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    #ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=outputchannels, 
        activation='sigmoid',
    )
    return model

class DLV3Reg(nn.Module):
    def __init__(self, segmentator, 
                 generator='/home/kafkaon1/FVAPP/third_party/projectRegularization/saved_models_gan/E140000_net', 
                 encoder='/home/kafkaon1/FVAPP/third_party/projectRegularization/saved_models_gan/E140000_e1'):
        super(DLV3Reg, self).__init__()
        self.modelSeg = torch.load(segmentator)

        self.encReg = Encoder()
        self.genReg = GeneratorResNet()
        self.genReg.load_state_dict(torch.load(generator))#, map_location=torch.device('cuda')))
        self.encReg.load_state_dict(torch.load(encoder))# , map_location=torch.device('cuda')))

    def forward(self, input):
        seg = self.modelSeg(input)
        print(type(seg.type(torch.uint8)))

        reg = []
        for i in range(seg.shape[0]):
            seg_i = seg[i,0,:,:].detach().cpu().numpy()
            in_i = input[i,:,:,:].detach().cpu().permute(1,2,0).numpy()
            reg_i = regularization(in_i, seg_i, [self.encReg, self.genReg])
            reg.append(torch.tensor(reg_i.astype(np.uint8)))

        return torch.stack(reg)
    
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


if __name__ == "__main__":
    img_path = '/home/kafkaon1/FVAPP/data/FV/train/image_resized/christchurch_450_478.png' #  '/home/kafkaon1/tmp_sataa.jpg'
    label_path = '/home/kafkaon1/FVAPP/data/FV/train/label_resized/christchurch_450_478.png'
    im = Image.open(img_path)
    la = np.array(Image.open(label_path)).astype(float)

    I = T.ToTensor()(im).to('cuda')#cv2.imread(img_path))
    l = T.ToTensor()(la)
    I = I.unsqueeze(0)
    l = l.unsqueeze(0)

    model = DLV3Reg('/home/kafkaon1/FVAPP/out/train/run_230522-093052/checkpoints/Deeplabv3_err:0.23320_ep:25.pth')
    model.eval()
    
    Ia = torch.vstack((I, I))
    outta = model(Ia)

    print(outta)
