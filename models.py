from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
import cv2
import numpy as np

import sys
sys.path.append("/home/kafkaon1/Dev/FVAPP/third_party")
from projectRegularization import GeneratorResNet,Encoder, regularization

sys.path.append("/home/kafkaon1/Dev/FVAPP/third_party/segment-anything/")
from segment_anything import sam_model_registry, SamPredictor
from skimage import measure


sys.path.append("/home/kafkaon1/Dev/FVAPP/third_party/segment-anything/")
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
                 generator='/home/kafkaon1/Dev/FVAPP/third_party/projectRegularization/saved_models_gan/E140000_net', 
                 encoder='/home/kafkaon1/Dev/FVAPP/third_party/projectRegularization/saved_models_gan/E140000_e1',
                 sam = '/home/kafkaon1/Dev/FVAPP/third_party/segment-anything/wghs.pth',
                 sam_type = 'vit_h',
                 do_sam = False,
                 do_reg = True,
                 do_poly = False,
                 device = 'cuda:0'):
        super(DLV3Reg, self).__init__()
        self.modelSeg = torch.load(segmentator)
        self.modelSeg.to(device)

        self.do_reg = do_reg
        if self.do_reg:
            self.encReg = Encoder()
            self.genReg = GeneratorResNet()
            self.genReg.load_state_dict(torch.load(generator, map_location=torch.device('cuda:0')))
            self.encReg.load_state_dict(torch.load(encoder , map_location=torch.device('cuda:0')))
            self.encReg.to(device)
            self.genReg.to(device)
        
        self.do_sam = do_sam
        if self.do_sam:
            sam = sam_model_registry["vit_h"](checkpoint="/home/kafkaon1/Dev/FVAPP/third_party/segment-anything/wghs.pth")
            sam.to(device)
            self.samPred = SamPredictor(sam) 
        
        self.do_poly = do_poly

    def predict(self, input):
        seg = (self.modelSeg(input) > 0.5).float()

        if not self.do_reg and not self.do_sam and not self.do_poly:
            return seg
        reg = []
        for i in range(seg.shape[0]):
            seg_i = seg[i,0,:,:].detach().cpu().numpy()
            in_i = (input[i,:,:,:].detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            if self.do_sam:
                self.samPred.set_image(in_i)
                bbxs = self.getBBxs(seg_i)
                t_bbxs = self.samPred.transform.apply_boxes_torch(bbxs, in_i.shape[:2])

                masks, _, _ = self.samPred.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=t_bbxs,
                    multimask_output=False,
                )
                seg_i = masks.sum(axis=(0))[0].detach().cpu().numpy()

            if self.do_reg:
                reg_i = regularization(in_i, seg_i, [self.encReg, self.genReg])
            else:
                reg_i = seg_i

            if self.do_poly:
                polygons = extractPolygons(reg_i, 0.005)
                reg_i = labelFromPolygons(polygons, in_i.shape[:2])
            
            reg.append(torch.tensor(reg_i.astype(np.uint8)))
                
        return torch.stack(reg).unsqueeze(1).float()
    
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
    

class PolyWorld(nn.Module):
    def __init__(self):
        # Load network modules
        model = R2U_Net()
        model = model.cuda()
        model = model.train()

        head_ver = DetectionBranch()
        head_ver = head_ver.cuda()
        head_ver = head_ver.train()

        suppression = NonMaxSuppression()
        suppression = suppression.cuda()

        matching = OptimalMatching()
        matching = matching.cuda()
        matching = matching.train()

        # NOTE: The modules are set to .train() mode during inference to make sure that the BatchNorm layers 
        # rely on batch statistics rather than the mean and variance estimated during training. 
        # Experimentally, using batch stats makes the network perform better during inference.

        print("Loading pretrained model")
        model.load_state_dict(torch.load("./trained_weights/polyworld_backbone"))
        head_ver.load_state_dict(torch.load("./trained_weights/polyworld_seg_head"))
        matching.load_state_dict(torch.load("./trained_weights/polyworld_matching"))

    def predict(self, input):
        features = model(rgb)
        occupancy_grid = head_ver(features)

        _, graph_pressed = suppression(occupancy_grid)

        poly = matching.predict(rgb, features, graph_pressed) 



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
