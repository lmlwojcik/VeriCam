from torchvision.models import vit_b_16, vit_b_32, vit_l_16

import torch
from torch import nn
from torch.nn import functional as F

from .utils import find_model

# Workaround for the case where the torch ssl certificate expires
# Uncomment if necessary
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

def ViT_pretrained(cfg):
    if cfg['model_cfg'] == 'vit_b_16':
        vit = vit_b_16(weights=cfg['vit_weights'])
    elif cfg['model_cfg'] == 'vit_b_32':
        vit = vit_b_32(weights=cfg['vit_weights'])
    elif cfg['model_cfg'] == 'vit_l_16':
        vit = vit_l_16(weights=cfg['vit_weights'])

    if cfg['freeze']:
        for c in vit.children():
            c.requires_grad = False

    n_classes = cfg['n_classes']
    n_ft = vit.heads.head.in_features
    class_head = nn.Sequential(nn.Linear(n_ft, n_classes))
    vit.heads = class_head

    return vit

# def get_model_with_weights(cfg, load_model, device, n_classes=4):
#     if cfg['model_name'] == 'small':
#         model = create_baseline(cfg['model_cfg'], cfg['n_features'], cfg['n_classes'])
#     elif cfg['model_name'] == 'resnet':
#         model = create_resnet(cfg, n_classes=n_classes)
#     elif cfg['model_name'] == 'yolo':
#         model = create_yolo(cfg, n_classes=n_classes, torch_training=True)
#     else:
#         model = create_vit(cfg, n_classes=n_classes)

#     if load_model is not None:
#         ckpt = torch.load(load_model)
#     else:
#         ckpt = torch.load(find_model(cfg['save_path']))
#     model.load_state_dict(ckpt)

#     if device != -1:
#         if len(device) == 1:
#             device = f"cuda:{device}"
#         model.to(torch.device(f"{device}"))
#     return model