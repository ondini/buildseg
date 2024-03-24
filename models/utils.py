import torch
import numpy as np

def reshape_param_weights(weights, tgt_shape, append = True):
    # thise funciton basically does nn.Module.expand() but pad with zeros and not repeat and newly allocated memory
    expanded = torch.zeros(tgt_shape)
    if not append: 
        idxs = weights.shape
        expanded[:idxs[0],:idxs[1],:idxs[2],:idxs[3]] = weights
    else:
        idxs = np.array(tgt_shape) - np.array(weights.shape)
        expanded[idxs[0]:,idxs[1]:,idxs[2]:,idxs[3]:] = weights
    return expanded

def load_ckpt_from_config(model, config, device='cpu'):
    """
    Load model weights from a checkpoint file, reshaping and ignoring parameters as specified in the config.
    """
    ckpt = torch.load(config['path'], map_location=device)
    unfreeze = set()
    if 'reshape_params' in config:
        for param_key, param_shape, append in config['reshape_params']:
            # unfreeze.add(param_key)
            ckpt[param_key] = reshape_param_weights(ckpt[param_key], param_shape, append)
    if 'ignore_params' in config:
        for param_key in config['ignore_params']:
            unfreeze.add(param_key)
            del ckpt[param_key]

    model.load_state_dict(ckpt, strict=config['strict'])

    for name, param in model.named_parameters():
        if name not in unfreeze:
            param.requires_grad = False
            

    print(f'Loaded model from {config["path"]}, unfroze {len(unfreeze)} parameters')

    return model