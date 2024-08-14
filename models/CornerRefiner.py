
import torch.nn as nn
import torch
import numpy as np
from .MaxVit import MaxVitUnetS


PATCH_SIZE = 224
SAMPLE_NEIGHBORS = 0
SAMPLE_DIST = 20
RETURN_UNCERTAIN_PTS = False

class CornerRefiner(nn.Module):
    def __init__(self, ckpt_path, device='cuda:0', noise_thr=0.08, conf_thr=0.12, dst_thr=35):
        '''
            ckpt_path : path to the checkpoint of the model
            device : device to run the model on
            noise_thr : threshold under which everything in keypoint heatmap is regarder as noise
            conf_thr : threshold above which every heatmap local maximum is regarded as new keypoint
            dst_thr : threshold for the distance between the refined keypoint and the original point
        '''
        super(CornerRefiner, self).__init__()
        self.model = MaxVitUnetS(logits=False)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.noise_thr = noise_thr
        self.conf_thr = conf_thr
        self.dst_thr = dst_thr
        
    
    def forward(self, img, roof_polygon):
        '''
            img : np.array of shape (H, W, 3)
            roof_polygon : polygonized roof mask to be refined
            
            returns : refined polygon
        '''
        thr = 0.1
        out_polygon = []
        heatmap_fin = np.zeros((img.shape[0], img.shape[1]))
        for pt in roof_polygon:
            heatmap_pt = np.zeros((img.shape[0], img.shape[1]))
            patch, offset, x_, y_ = get_patch(img, *pt) # get patch around the point and the offset of the point in the patch
            patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()/255
            out = self.model(patch)
            
            heatmap_pt[y_-PATCH_SIZE//2:y_+PATCH_SIZE//2, x_-PATCH_SIZE//2:x_+PATCH_SIZE//2] = out[0][0].cpu().detach().numpy()
            heatmap_fin = np.maximum(heatmap_fin, heatmap_pt)

            for i in range(SAMPLE_NEIGHBORS):
                angle = 0 + i * 2*np.pi/SAMPLE_NEIGHBORS
                x = pt[0] + SAMPLE_DIST*np.cos(angle)
                y = pt[1] + SAMPLE_DIST*np.sin(angle)
                heatmap_pt = np.zeros((img.shape[0], img.shape[1]))
                patch, offset, x_, y_ = get_patch(img, x, y) # get patch around the point and the offset of the point in the patch
                patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()/255
                out = self.model(patch)
                
                heatmap_pt[y_-PATCH_SIZE//2:y_+PATCH_SIZE//2, x_-PATCH_SIZE//2:x_+PATCH_SIZE//2] = out[0][0].cpu().detach().numpy()
                heatmap_fin = np.maximum(heatmap_fin, heatmap_pt)
            
        htt = torch.from_numpy(heatmap_fin).unsqueeze(0).unsqueeze(0).float()
        sc, indi = get_keypoint_coords(htt, self.noise_thr)
        ind = indi[0] # as there is no batch approach imlpemented yet
        if len(ind) == 0:
            return roof_polygon, heatmap_fin
        
        for pt in roof_polygon:
            dists = torch.pow(ind - np.array([pt[1], pt[0]]), 2).sum(axis=1)
            print(dists)         
            # closest keypoint
            dst = dists.min().item()
            dst_id = dists.argmin().item()
            
            score = sc[0][dst_id]
            if score > self.conf_thr and dst < self.dst_thr**2:
                print('good point')
                indc = ind[torch.argmin(dists)]
                out_polygon.append((indc[1], indc[0]))
            elif RETURN_UNCERTAIN_PTS:
                print('uncertain point')
                out_polygon.append(pt)
            
        return out_polygon
            
    def forward_old(self, img, roof_polygon):
        '''
            img : np.array of shape (H, W, 3)
            roof_polygon : polygonized roof mask to be refined
            
            returns : refined polygon
        '''
        thr = 0.1
        out_polygon = []
        for pt in roof_polygon:
            
            patch, offset = get_patch(img, *pt) # get patch around the point and the offset of the point in the patch
            patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()/255
            out = self.model(patch)
            
            sc, indi = get_keypoint_coords(out, self.noise_thr)
            ind = indi[0] # as there is no batch approach imlpemented yet
            if len(ind) == 0:
                continue
            dists = torch.pow(ind - np.array(offset), 2).sum(axis=1)
            print(dists)         
            # closest keypoint
            dst = dists.min().item()
            dst_id = dists.argmin().item()
            
            score = sc[0][dst_id]
            if score > thr and dst < self.dst_thr**2:
                indc = ind[torch.argmin(dists)]
                out_polygon.append(indc)
            
        return out_polygon
        
def get_patch(img, x, y):
    # create 256x256 patch with the point in the middle from img with prevence against overflowing
    
    x_ = x if x >= PATCH_SIZE//2 else PATCH_SIZE//2
    y_ = y if y >= PATCH_SIZE//2 else PATCH_SIZE//2
    x_ = x_ if x_ <= img.shape[1] - PATCH_SIZE//2 -1 else img.shape[1] - PATCH_SIZE//2 - 1
    y_ = y_ if y_ <= img.shape[0] - PATCH_SIZE//2 -1 else img.shape[0] - PATCH_SIZE//2 - 1
    x_ = int(x_)
    y_ = int(y_)
    
    # offset of the point in the patch
    xo = x - x_ + PATCH_SIZE//2
    yo = y - y_ + PATCH_SIZE//2
    return img[y_-PATCH_SIZE//2:y_+PATCH_SIZE//2, x_-PATCH_SIZE//2:x_+PATCH_SIZE//2], torch.tensor([yo, xo]),  x_, y_


def get_keypoint_coords(heatmap, noise_thr):
    '''
        Function for getting the keypoint coordinates from the heatmap,
        it is basically getting local maxima of the heatmap using maxpooling
        TODO - complete batch-based approach
    '''
    min_keypoint_pixel_distance = 5
    
    batch_size, n_channels, _, width = heatmap.shape
    heatmap[heatmap < noise_thr] = 0 # filter noise
    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = min_keypoint_pixel_distance
    # exclude border keypoints by padding with highest possible value
    # bc the borders are more susceptible to noise and could result in false positives
    padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=0)
    max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)
    # if the value equals the original value, it is the local maximum
    local_maxima = torch.bitwise_and(max_pooled_heatmap == heatmap, heatmap > 0)
    # all values to zero that are not local maxima
    heatmap = heatmap * local_maxima
    num_pts = torch.sum(local_maxima)
    if num_pts == 0: # no local maxima found - topk would complain
        return torch.zeros(batch_size, 0, 1), torch.zeros(batch_size, 0, 2)
    
    # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
    scores, indices = torch.topk(heatmap.view(batch_size, -1), num_pts, sorted=True)
    indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
    return scores, indices

