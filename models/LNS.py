import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndimage
import numpy as np
import cv2

class CompressionStage(nn.Module):
    """(Conv => BN => ReLU) * 3"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, use_bias=True):
        super().__init__()
        padding = (kernel_size-1)//2 # "same"
        self.tripleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=use_bias), # add Separable Conv2D?? 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.tripleconv(x)


class DecompressionStage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_bias=True):
        super().__init__()
        
        padding = (kernel_size-1)//2 # "same"
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, x1, x2):
        x1 = self.deconv1(x1)
        y = x1 + x2
        return self.deconv2(y)


class LN2(nn.Module):
    def __init__(self, in_channels, num_classes, use_bias=True, use_horizon=False, iim=False, obj_thr=0.6):
        super().__init__()
        
        self.use_horizon = use_horizon
        self.use_bias = use_bias
        self.num_classes = num_classes
        self.iim = iim
        self.obj_thr = obj_thr
        
        # Create the image input layer
        in_channels = in_channels + self.iim*in_channels
        
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Create the encoder part of the network
        self.cp_st1 = CompressionStage(32, 64, use_bias=self.use_bias)
        self.cp_st2 = CompressionStage(64, 64, use_bias=self.use_bias)
        self.cp_st3 = CompressionStage(64, 128, use_bias=self.use_bias)
        self.cp_st4 = CompressionStage(128, 256, use_bias=self.use_bias)

        # Create the decoder part of the network
        self.dcp_st1 = DecompressionStage(256, 128, use_bias=self.use_bias)
        self.dcp_st2 = DecompressionStage(128, 64, use_bias=self.use_bias)
        self.dcp_st3 = DecompressionStage(64, 64, use_bias=self.use_bias)

        
        self.outpu_layer = nn.Sequential(
            nn.Conv2d(64, self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1)
        )
        
        if self.use_horizon:
            self.output_layer_horizon = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        if self.iim:
            x = torch.cat((x, x), 1)
            
        x = self.inconv(x)
        x_cp1 = self.cp_st1(x)
        x_cp2 = self.cp_st2(x_cp1)
        x_cp3 = self.cp_st3(x_cp2)
        x_cp4 = self.cp_st4(x_cp3)
        
        x_dcp1 = self.dcp_st1(x_cp4, x_cp3)
        x_dcp2 = self.dcp_st2(x_dcp1, x_cp2)
        x_dcp3 = self.dcp_st3(x_dcp2, x_cp1)
        
        x = self.outpu_layer(x_dcp3)
        out = {"cm": x}
        
        if self.use_horizon:
            x_hm = self.output_layer_horizon(x_dcp3)
            out["hm"] = x_hm
        
        return out
    
    def process_bbxs(self, x, obj_th):
        out = {**x}
        batch_size = x['cm'].shape[0]
        rects_l = [] 
        horizon_l = []
        for i in range(batch_size):
            print(x[i,:,:,:].permute(1,2,0).detach().numpy())
            rects = self.CNN_out_to_object_detection(x[i,:,:,:].permute(1,2,0).detach().numpy(), obj_th=obj_th)
            rects_l.append(rects)

            # horizon = self.CNN_out_to_horizon_detections(r["hm"][i,:,:,0], horizonw_detection_parameter)
            # horizon_l.append(horizon)
        
        out["rect"] = rects_l
        #out["horizon"] = horizon_l
        return out
    
    def CNN_out_to_object_detection(self, class_detections, obj_th=0.5):
        """Obtain bounding boxes and classification of detections.
        Argument:
        class_detections -- 3D array (image_height, image_width, num_channels)
        Returns:
        list of rectangles with classification vector.
        First apply a softmax normalization to the output of the CNN.
        The zeroth channel is then the probability of the pixel belonging
        to the background and therefore used to generate a blob mask
        by ndimage.label.
        For the blobs we generate bounding boxes by ndimage.find_objects.
        Then we iterate over all detected objects and generate the output list:
        Each element contains the normalized coordinates of the rectangle corners
        and the mean classification probability, where the mean is taken not over
        the whole rectangle, but only over the object shape.
        """
    
        # pixels that are "probably" not background
        det_model_blob_mask = (class_detections[:, :, 0] > obj_th).astype(np.uint8)
        # det_model_blob_mask = class_detections[:, :, 0] != np.amax(class_detections, axis=2)
    
        # detect blobs and bounding boxes (defined by slices)
        conn = np.array([[1,1,1],[1,1,1],[1,1,1]])
        obj_map_labeld, number_of_blobs, = ndimage.label(det_model_blob_mask, structure=conn)
        object_slices = ndimage.find_objects(obj_map_labeld)
        print(obj_map_labeld.shape)
        img_dims = obj_map_labeld.shape
        rects = []
        for i in range(number_of_blobs):
            my_slice = object_slices[i]
    
            min_row, max_row = my_slice[0].start, my_slice[0].stop
            min_col, max_col = my_slice[1].start, my_slice[1].stop
    
            # from the whole arrays, cut out the rectangle of interest
            object_cutout = class_detections[my_slice]
            mask_cutout = obj_map_labeld[my_slice]
    
    
            # spatial average over the object -> vector in object category space
            class_pred = np.mean(object_cutout[mask_cutout == i + 1, :], axis=0)
            # y,x = object_cutout.shape[:2]
            # class_pred = object_cutout[y//2,x//2,:]
    
            #rect = [min_col / img_dims[1], min_row / img_dims[0],max_col / img_dims[1], max_row / img_dims[0],class_pred]
            rect = [min_col , min_row, max_col, max_row , class_pred]
    
            rects.append(rect)
    
        return rects