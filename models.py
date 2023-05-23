from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

import sys
sys.path.append("./third_party")
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
        self.genReg.load_state_dict(torch.load(generator))
        self.encReg.load_state_dict(torch.load(encoder))

    def forward(self, input):
        seg = self.modelSeg(input)
        print(type(seg.type(torch.uint8)))

        reg = []
        for i in range(len(seg.shape[0])):
            seg_i = seg[i,0,:,:].detach()
            in_i = input[i,:,:,:].detach()
            reg_i = regularization(in_i, seg_i, [self.encReg, self.genReg])
            reg.append(reg_i)
            
        return reg
