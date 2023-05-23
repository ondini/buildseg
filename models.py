from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

import segmentation_models_pytorch as smp

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

