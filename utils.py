import cv2
from natsort import natsorted
from glob import glob
import json
from pathlib import Path

def dict_to_string(dct):
    # For logging purposes
    ret = ""
    for k, v in dct.items():
        ret += f"| {k}: {v}"
    return ret

def dict_to_table(dct):
    keys = ['epoch', 'train_loss', 'val_acc', 'val_f1']
    ret = ""
    #for k in dct.keys():
    for k in keys:
        ret += f" {k:<15}|"
    ret += "\n"
    #ret += f"{'-'*(16*len(dct.keys()) + len(dct.keys()))}"
    ret += f"{'-'*(16*len(keys) + len(keys))}"
    ret += "\n"
    #for v in dct.values():
    #    ret += f" {v:<15.4g}|"
    for k in keys:
        ret += f" {dct[k]:<15.4g}|"
    return ret

def find_model(save_path):
    # By the syntax the outputted models, the last one in the list is the latest
    models = natsorted(list(glob(f"{save_path}/*.pth")))
    for m in models:
        if "best" in m:
            print("Found best model ", m)
            return m
    chosen = m
    print("Found last model ", m)
    return chosen

def start_log(log_file="logs/log.log"):
    with open(log_file, "w") as fd:
        fd.write("[")

def log_metrics_json(metrics, log_file="logs/log.log"):
    with open(log_file, "a") as fd:
        fd.write(json.dumps(metrics) + ",\n")

def end_log(log_file="logs/log.log"):
    with open(log_file, "a") as fd:
        fd.write("\{\}]")

def log_config(save_path, cfg):
    with open(save_path / Path("experiment_config.json"), "w") as fd:
        json.dump(cfg, fd, indent=2)

def resize_with_pad(image, 
                    new_shape, 
                    padding_color = (127, 127, 127)):
    h, w = image.shape[:2]
    nh,nw = new_shape[:2]
    
    sw = nw / w
    sh = nh / h
    scale = min(sw, sh)

    if w > nw or h > nh:
        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w = w
        new_h = h

    delta_w = nw - new_w
    delta_h = nh - new_h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

