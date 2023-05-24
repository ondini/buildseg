from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

import sys
sys.path.append("/home/kafkaon1/FVAPP/third_party")
from projectRegularization import GeneratorResNet,Encoder, regularization

sys.path.append("/home/kafkaon1/FVAPP/third_party/segment-anything/")
from segment_anything import sam_model_registry, SamPredictor
from skimage import measure

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
                 encoder='/home/kafkaon1/FVAPP/third_party/projectRegularization/saved_models_gan/E140000_e1',
                 sam = '/home/kafkaon1/FVAPP/third_party/segment-anything/wghs.pth',
                 sam_type = 'vit_h',
                 do_sam = True,
                 do_reg = True):
        super(DLV3Reg, self).__init__()
        self.modelSeg = torch.load(segmentator)

        self.do_reg = do_reg
        if self.do_reg:
            self.encReg = Encoder()
            self.genReg = GeneratorResNet()
            self.genReg.load_state_dict(torch.load(generator))#, map_location=torch.device('cuda')))
            self.encReg.load_state_dict(torch.load(encoder))# , map_location=torch.device('cuda')))
        
        self.do_sam = do_sam
        if self.do_sam:
            sam = sam_model_registry["vit_h"](checkpoint="/home/kafkaon1/FVAPP/third_party/segment-anything/wghs.pth")
            sam.to('cuda:1')
            self.samPred = SamPredictor(sam) 

    def forward(self, input):
        seg = self.modelSeg(input)
        print(type(seg.type(torch.uint8)))

        reg = []
        for i in range(seg.shape[0]):
            seg_i = seg[i,0,:,:].detach().cpu().numpy()
            in_i = input[i,:,:,:].detach().cpu().permute(1,2,0).numpy()
            if self.do_sam:
                self.samPred.set_image((in_i*255).astype(np.uint8))
                bbxs = self.getBBxs(seg_i)
                t_bbxs = self.samPred.transform.apply_boxes_torch(bbxs, in_i.shape[:2])

                masks, _, _ = self.samPred.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=t_bbxs,
                    multimask_output=False,
                )
                seg_i = masks.sum(axis=(0))[0].detach().cpu().numpy()

            reg_i = regularization(in_i, seg_i, [self.encReg, self.genReg])
            reg.append(torch.tensor(reg_i.astype(np.uint8)))

        return torch.stack(reg)
    
    def getBBxs(self, ins_segmentation):
        ins_segmentation = np.uint16(measure.label(ins_segmentation, background=0))

        max_instance = np.amax(ins_segmentation)
        min_size=10

        bbxs = []
        for ins in range(1, max_instance+1):
            indices = np.argwhere(ins_segmentation==ins)
            building_size = indices.shape[0]
            if building_size > min_size:
                i_min = np.amin(indices[:,0])
                i_max = np.amax(indices[:,0])
                j_min = np.amin(indices[:,1])
                j_max = np.amax(indices[:,1])

                label_bbx = [j_min, i_min, j_max, i_max]
                bbxs.append(label_bbx)

        bbxs = torch.tensor(bbxs, device=self.samPred.device)   
        return bbxs
    
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
    model.to('cuda:1')
    
    Ia = torch.vstack((I, I)).to('cuda:1')
    outta = model(Ia)

    print(outta)
