import torch.nn as nn
import torch
import numpy as np

from .MaxVit import MaxVitUnet

class CornerRefiner(nn.Module):
    def __init__(self, ckpt_path, device='cuda:0', noise_thr=0.08, conf_thr=0.18, dst_thr=30):
        '''
            ckpt_path : path to the checkpoint of the model
            device : device to run the model on
            noise_thr : threshold under which everything in keypoint heatmap is regarder as noise
            conf_thr : threshold above which every heatmap local maximum is regarded as new keypoint
            dst_thr : threshold for the distance between the refined keypoint and the original point
        '''
        super(CornerRefiner, self).__init__()
        self.model = MaxVitUnet(logits=False)
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
        for pt in roof_polygon:
            
            patch, offset = get_patch(img, *pt) # get patch around the point and the offset of the point in the patch
            patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()/255
            out = self.model(patch)
            
            sc, indi = get_keypoint_coords(out, self.noise_thr)
            ind = indi[0] # as there is no batch approach imlpemented yet
            print(offset, ind, sc)
            plt.imshow(patch[0].permute(1,2,0).cpu().detach().numpy())
            plt.imshow(out[0][0].cpu().detach().numpy(), alpha=0.3)
            plt.scatter(ind[:, 1], ind[:, 0], c='r', s=1)
            plt.show()
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
    x_ = x_ if x_ <= img.shape[1] - PATCH_SIZE//2 else img.shape[1] - PATCH_SIZE//2
    y_ = y_ if y_ <= img.shape[0] - PATCH_SIZE//2 else img.shape[0] - PATCH_SIZE//2
    x_ = int(x_)
    y_ = int(y_)
    
    # offset of the point in the patch
    xo = x - x_ + PATCH_SIZE//2
    yo = y - y_ + PATCH_SIZE//2
    return img[y_-PATCH_SIZE//2:y_+PATCH_SIZE//2, x_-PATCH_SIZE//2:x_+PATCH_SIZE//2], torch.tensor([yo, xo])


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
