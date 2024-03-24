import os
import datetime
import argparse
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader

from models import DLV3Reg, extractPolygons, labelFromPolygons
from dataset import FVDataset
from metrics import GetMetricFunction, Metrics
from configs import model_weights

def evaluate(args):
    start = datetime.datetime.now()
    run_name = f"run_{start:%y%m%d-%H%M%S}"
    out_path = os.path.join(args.out_path, run_name)

    os.makedirs(out_path)

    val_data_path = os.path.join(args.dataset_path, 'val/image_resized')
    val_label_path = os.path.join(args.dataset_path, 'val/label_resized')
    val_list_path = os.path.join(args.dataset_path, 'val.txt')

    logging.basicConfig(filename=os.path.join(out_path, 'logging.log'), level=logging.INFO)

    valid_dataset = FVDataset(
        val_data_path, val_label_path, val_list_path,
        augmentation=False,
        size_coefficient=args.dataset_coefficient,
    )

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results_final = {}
    for model_name in model_weights.keys():
        model_path = model_weights[model_name]
        if args.use_reg or args.use_sam or args.get_poly:
            model = DLV3Reg(model_path, do_reg=args.use_reg, do_sam=args.use_sam, do_poly=args.get_poly)
        else:
            model = torch.load(model_path)

        model.to(device)
        model.eval()

        desc = f'FV: {run_name} \n \
            Model: {model_name} \n \
            Checkpoint: {model_path} \n \
            Batch size: {args.batch_size} \n \
            Use reg: {args.use_reg} \n \
            Use sam: {args.use_sam} \n \
            Get poly: {args.get_poly} \n \
            Hard pred: {args.hard_pred} \n \
            Dataset coefficient: {args.dataset_coefficient} \n \
            Device: {device} \n \n'

        logging.info(desc)

        y_true, y_pred = [], []

        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model.predict(inputs)

            y_true.append(targets.cpu().float())
            if args.hard_pred:
                y_pred.append((predictions.cpu()>0.5).float())
            else:
                y_pred.append(predictions.cpu().float())

        
        y_true, y_pred = torch.concat(y_true,0), torch.concat(y_pred, 0)

        metric_names = ["iou", "dice"] #, "matthews", "precision", "recall"]
        metrics = {
            metric_name : GetMetricFunction(metric_name) for metric_name in metric_names
        } 
        metrics.update( {
            f"{metric_name}_b{1}" : GetMetricFunction(metric_name, 1) for metric_name in metric_names[1:]
        })
        metrics.update({
            f"{metric_name}_b{3}" : GetMetricFunction(metric_name, 3) for metric_name in metric_names[1:]
        } )
        metrics.update({
            f"{metric_name}_b{5}" : GetMetricFunction(metric_name, 5) for metric_name in metric_names[1:]
        } )
        
        results = {}
        for metric_name, metric in metrics.items():
            res = metric(y_true, y_pred)
            logging.info(f'{metric_name}: {res.mean()}')
            results[metric_name] = res
        
        results_final[model_name] = results
        
    np.save(os.path.join(out_path,"results.npy"), results_final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FVApp evaluation script')
    parser.add_argument("--device", type=str,  choices={'cuda:0', 'cuda:1', 'cpu'}, default='cuda:0')
    parser.add_argument("--out_path", type=str, default='/home/kafkaon1/FVAPP/out/eval')
    parser.add_argument("--dataset_path", type=str, default='/home/kafkaon1/FVAPP/data/FV')
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--use_reg", type=bool, default=False)
    parser.add_argument("--use_sam", type=bool, default=False)
    parser.add_argument("--hard_pred", type=bool, default=False)
    parser.add_argument("--get_poly", type=bool, default=True)
    parser.add_argument("--dataset_coefficient", type=float, default=1/50)

    args = parser.parse_args()

    evaluate(args)
    
    
    
def get_keypoints_from_heatmap_batch_maxpool(
    heatmap: torch.Tensor,
    max_keypoints: int = 20,
    min_keypoint_pixel_distance: int = 1,
    abs_max_threshold: Optional[float] = None,
    rel_max_threshold: Optional[float] = None,
    return_scores: bool = False,
) -> List[List[List[Tuple[int, int]]]]:
    """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

    Inspired by mmdetection and CenterNet:
      https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html

    Args:
        heatmap (torch.Tensor): NxCxHxW heatmap batch
        max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
        min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

        Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
        abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
        rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

    Returns:
        The extracted keypoints for each batch, channel and heatmap; and their scores
    """

    # TODO: maybe separate the thresholding into another function to make sure it is not used during training, where it should not be used?

    # TODO: ugly that the output can change based on a flag.. should always return scores and discard them when I don't need them...

    batch_size, n_channels, _, width = heatmap.shape

    # obtain max_keypoints local maxima for each channel (w/ maxpool)

    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = min_keypoint_pixel_distance
    # exclude border keypoints by padding with highest possible value
    # bc the borders are more susceptible to noise and could result in false positives
    padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
    max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)
    # if the value equals the original value, it is the local maximum
    local_maxima = max_pooled_heatmap == heatmap
    # all values to zero that are not local maxima
    heatmap = heatmap * local_maxima

    # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
    scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
    indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
    # at this point either score > 0.0, in which case the index is a local maximum
    # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

    #  remove top-k that are not local maxima and threshold (if required)
    # thresholding shouldn't be done during training

    #  moving them to CPU now to avoid multiple GPU-mem accesses!
    indices = indices.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    # determine NMS threshold
    threshold = 0.01  # make sure it is > 0 to filter out top-k that are not local maxima
    if abs_max_threshold is not None:
        threshold = max(threshold, abs_max_threshold)
    if rel_max_threshold is not None:
        threshold = max(threshold, rel_max_threshold * heatmap.max())

    # have to do this manually as the number of maxima for each channel can be different
    for batch_idx in range(batch_size):
        for channel_idx in range(n_channels):
            candidates = indices[batch_idx, channel_idx]
            for candidate_idx in range(candidates.shape[0]):

                # these are filtered out directly.
                if scores[batch_idx, channel_idx, candidate_idx] > threshold:
                    # convert to (u,v)
                    filtered_indices[batch_idx][channel_idx].append(candidates[candidate_idx][::-1].tolist())
                    filtered_scores[batch_idx][channel_idx].append(scores[batch_idx, channel_idx, candidate_idx])
    if return_scores:
        return filtered_indices, filtered_scores
    else:
        return filtered_indices