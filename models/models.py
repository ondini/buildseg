import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models.detection as det
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from third_party import GeneratorResNet,Encoder,regularization
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, n_classes, n_channels=4):
        super().__init__()
        self.model = fcn_resnet50(pretrained=False, num_classes=n_classes, aux_loss=None)
        
    def forward(self, x):
        return self.model(x)
    
class DeeplabV3(nn.Module):
    def __init__(self, num_classes, n_channels=4):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=None, num_classes=num_classes, aux_loss=None)
        
    def forward(self, x):
        return F.sigmoid(self.model(x)['out'])


class DeepLabv3Plus(nn.Module):
    """ DeepLabv3Plus - model for semantic segmentation with resnet50 backbone
    # Args
        outputchannels: number of classes
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
    """
    def __init__(self, num_classes, logits=False, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        ACTIVATION = 'sigmoid' if not logits else 'logits' # could be None for logits or 'softmax2d' for multiclass segmentation
        # create segmentation model with pretrained encoder
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            classes=num_classes, 
            activation=ACTIVATION,
        )
        self.logits = logits
    
    def forward(self, x):
        return self.model(x)


class MaskRCNN(nn.Module):
    """ MaskRCNN - model for object detection with resnet50 backbone
    # Args
        num_classes: number of classes
        fast: if True, use fasterrcnn_resnet50_fpn, else use maskrcnn_resnet50_fpn_v2  
    """
    
    def __init__(self, num_classes, fast=False, loss='default'):
        from scoring import make_fasrcnn_loss
        
        super().__init__()
        self.num_classes = num_classes
        if not fast:
            self.model = det.maskrcnn_resnet50_fpn(pretrained=True)
        else:
            self.model = det.fasterrcnn_resnet50_fpn(pretrained=True)

        det.roi_heads.maskrcnn_loss = make_fasrcnn_loss('CombLoss', fast)
        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        for module in self.model.modules():
            if 'output_size' in dir(module) and module.output_size == (14,14):
                module.output_size = (56,56)
                
    def forward(self, x):
        return self.model(x)
    
class Regularizer(nn.Module):
    def __init__(self, device='cuda:0',
                 generator='/home/kafkaon1/Dev/FVAPP/third_party/PR/saved_models_gan/E95000_net', 
                 encoder='/home/kafkaon1/Dev/FVAPP/third_party/PR/saved_models_gan/E95000_e1',):
        super().__init__()
        self.encReg = Encoder()
        self.genReg = GeneratorResNet()
        self.genReg.load_state_dict(torch.load(generator, map_location=torch.device(device)))
        self.encReg.load_state_dict(torch.load(encoder , map_location=torch.device(device)))
        self.encReg.to(device)
        self.genReg.to(device)
        
    def forward(self, img, seg):
        return regularization(img, seg, [self.encReg, self.genReg])
    
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules


class RefineMask(nn.Module):
    """ RefineMask - model for instance segmentation with resnet50 backbone"""
    def __init__(self, cfg_path, ckpt_path=None):
        super().__init__()
        register_all_modules()
        if ckpt_path is not None:
            self.model = init_detector(cfg_path, ckpt_path)
        else:
            self.model = init_detector(cfg_path)

    def forward(self, x):
        return inference_detector(self.model, x)