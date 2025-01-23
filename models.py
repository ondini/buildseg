from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
import cv2
import numpy as np
from skimage import measure

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
    def __init__(self, do_reg=True, do_poly=True):
        super(DLV3Reg, self).__init__()

        self.modelSeg = smp.DeepLabV3Plus(
            encoder_name='resnet50',
            activation='sigmoid',
        ) 
        
        self.do_poly = do_poly  
        self.do_reg = do_reg
        if self.do_reg:
            self.encReg = Encoder()
            self.genReg = GeneratorResNet()
                    
    
    def load_weights(self, segmentator,
                 generator='/home/kafkaon1/Dev/FVAPP/third_party/projectRegularization/saved_models_gan/E140000_net', 
                 encoder='/home/kafkaon1/Dev/FVAPP/third_party/projectRegularization/saved_models_gan/E140000_e1',
                 device='cuda:0'):
        self.modelSeg.load_state_dict(torch.load(segmentator))
        self.modelSeg.to(device)

        if self.do_reg:
            self.genReg.load_state_dict(torch.load(generator))
            self.encReg.load_state_dict(torch.load(encoder))
            self.encReg.to(device)
            self.genReg.to(device)

    def predict(self, input):
        seg = (self.modelSeg(input) > 0.5).float()

        if not self.do_reg and not self.do_sam and not self.do_poly:
            return seg
        reg = []
        poly = [] if self.do_poly else None
        for i in range(seg.shape[0]):
            seg_i = seg[i,0,:,:].detach().cpu().numpy()
            in_i = (input[i,:,:,:].detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)

            if self.do_reg:
                reg_i = regularization(in_i, seg_i, [self.encReg, self.genReg])
            else:
                reg_i = seg_i

            if self.do_poly:
                polygons = extractPolygons(reg_i, 0.005)
                reg_i = labelFromPolygons(polygons, in_i.shape[:2])
                poly.append(polygons)
            
            reg.append(torch.tensor(reg_i.astype(np.uint8)))
                
        return torch.stack(reg).unsqueeze(1).float(), poly
    
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
    

def extractPolygons(segmentation, eps =  0.01):
    segmentation = np.uint16(measure.label(segmentation, background=0))

    max_instance = np.amax(segmentation)
    min_size=500

    polygons = []
    for ins in range(1, max_instance+1):
        shape_mask = np.uint8(segmentation == ins)

        if shape_mask.sum() > min_size:
            contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = eps * cv2.arcLength(largest_contour, True)

            approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

            vertices = [tuple(vertex[0]) for vertex in approx_polygon]
            polygons.append(vertices)

    return polygons

def labelFromPolygons(polygons, shape):
    label = np.zeros(shape, dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(label, [np.array(poly)], 1)
    return label


import torchvision.transforms as T
from PIL import Image

if __name__ == "__main__":
    img_path = '/home/kafkaon1/FVAPP/data/FV/train/image_resized/christchurch_450_478.png' #  '/home/kafkaon1/tmp_sataa.jpg'
    label_path = '/home/kafkaon1/FVAPP/data/FV/train/label_resized/christchurch_450_478.png'
    im = Image.open(img_path)
    la = np.array(Image.open(label_path)).astype(float)

    I = T.ToTensor()(im).to('cuda')#cv2.imread(img_path))
    l = T.ToTensor()(la)
    I = I.unsqueeze(0)
    l = l.unsqueeze(0)

    model = DLV3Reg('/home/kafkaon1/FVAPP/out/train/run_230522-093052/checkpoints/Deeplabv3_err:0.23320_ep:25.pth', do_reg=False, do_poly=True)
    model.eval()
    model.to('cuda:1')
    
    Ia = torch.vstack((I, I)).to('cuda:1')
    outta = model.predict(Ia)


    print(outta)
