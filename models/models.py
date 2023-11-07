import segmentation_models_pytorch as smp
import torch.nn as nn
import torchvision.models.detection as det
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class DeepLabv3Plus(nn.Module):
    """ DeepLabv3Plus - model for semantic segmentation with resnet50 backbone
    # Args
        outputchannels: number of classes
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
    """
    def __init__(self, num_classes, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
        # create segmentation model with pretrained encoder
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            classes=num_classes, 
            activation=ACTIVATION,
        )
    
    def forward(self, x):
        return self.model(x)

class MaskRCNN(nn.Module):
    """ MaskRCNN - model for object detection with resnet50 backbone
    # Args
        num_classes: number of classes
        fast: if True, use fasterrcnn_resnet50_fpn, else use maskrcnn_resnet50_fpn_v2  
    """
    
    def __init__(self, num_classes, fast=False, loss='default'):
        super().__init__()
        self.num_classes = num_classes
        if not fast:
            self.model = det.maskrcnn_resnet50_fpn(pretrained=True)
        else:
            self.model = det.fasterrcnn_resnet50_fpn(pretrained=True)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class Regularizer(nn.Module):
    def __init__(self, num_classes,
                 generator='/home/kafkaon1/Dev/FVAPP/third_party/projectRegularization/saved_models_gan/E140000_net', 
                 encoder='/home/kafkaon1/Dev/FVAPP/third_party/projectRegularization/saved_models_gan/E140000_e1',):
        super().__init__()
        self.encReg = Encoder()
        self.genReg = GeneratorResNet()
        self.genReg.load_state_dict(torch.load(generator, map_location=torch.device('cuda:0')))
        self.encReg.load_state_dict(torch.load(encoder , map_location=torch.device('cuda:0')))
        self.encReg.to(device)
        self.genReg.to(device)
    
    def forward(self, x):
        
        x = self.encReg(x)
        x = self.genReg(x)
        return x