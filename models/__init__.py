from glob import glob
import torch
from .vit import VisionTransformer

def create_model(cfg):
    if cfg['model_name'] == 'vit':
        return VisionTransformer(cfg['model_args'])
    else:
        raise ValueError("Model not supported or recognized: ", cfg['model_name'])

def get_model_with_weights(cfg, load_model, device, n_classes=512):
    model = create_model(cfg)

    if load_model is not None:
        ckpt = torch.load(load_model)
    else:
        ckpt = torch.load(find_model(cfg['save_path']))
    model.load_state_dict(ckpt)

    if device != -1:
        if len(device) == 1:
            device = f"cuda:{device}"
        model.to(torch.device(f"{device}"))
    return model