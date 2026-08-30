from torch.nn.functional import cosine_similarity
import torch

from glob import glob
import cv2
import os
from tqdm import tqdm
import argparse

from models import create_model, get_model_with_weights
from dataset import load_image

device = torch.device('cuda:0')
size = 224
img_size = [size, size]

cfg = {
    "model_name": "vit",
    "model_args": {
        "img_size": 224,
        "patch_size": 14,
        "in_channels": 3,
        "num_classes": 512,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "device": "cuda:0"
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_file', type=str)
    parser.add_argument('-i', '--images_dir', type=str)
    parser.add_argument('-o', '--output', type=str)

    args = vars(parser.parse_args())

    model = create_model(cfg)
    model = get_model_with_weights(model, args['model_file'], None, 'cuda:0')
    model.eval()
    tensors = []

    for f in tqdm(glob(f"{args['images_dir']}/*")):
        im = load_image(f, img_size, device)

        ft = model(im.unsqueeze(0))

        bn = os.path.splitext(os.path.basename(f))[0]
        torch.save(ft, f"{args['output']}/by_ims/{bn}.pt")

    tensors = torch.stack(tensors)
    torch.save(tensors, f"{args['output']}/all_fts_stacked.pt")
