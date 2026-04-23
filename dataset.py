from torch.utils.data import Dataset
import torch
import cv2

import random
import json
from time import time

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

def load_image(fname, size, device):
    im = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    im = torch.Tensor(resize_with_pad(im, size)).to(device)
    im = torch.permute(neg, (2, 0, 1))
    return im

class DynamicDataset(Dataset):
    def __init__(self, cam_info_file, n_samples, size=[256, 256],
                device='cpu', use_triplet=True, true_ratio=0.3, exclude_idx=[], seed=None):
        self.use_triplet = use_triplet
        self.true_ratio = true_ratio
        self.n_samples = n_samples
        self.device = device
        self.size = size

        if seed is None:
            seed = int(time.time())
        self.seed = seed
        # Seed for random is set on main, this is just for logging

        with open(cam_info_file, "r") as fd:
            ls = json.load(fd)
        for idx in exclude_idx:
            del ls[idx]
        
        self.classes = list(ls.keys())
        self.valid_classes = [x for x,v in ls.items() if len(v) >= 2]

        self.dataset = ls

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.use_triplet:
            anchor_class = random.choice(self.valid_classes)
            neg_class = anchor_class
            while neg_class == anchor_class:
                neg_class = random.choice(self.classes)

            anchor_file, pos_file = random.sample(self.dataset[anchor_class], k=2)
            neg_file = random.choice(self.dataset[neg_class])

            im = load_image(anchor_file, self.size, self.device)
            pos = load_image(pos_file, self.size, self.device)
            neg = load_image(neg_file, self.size, self.device)

            return anchor, pos, neg
        else:
            if random.random() > self.true_ratio:
                # Negative
                lb = torch.Tensor([0]).to(self.device).to(torch.long)

                chosen_classes = random.sample(self.classes, k=2)
                im1_file = random.choice(self.dataset[chosen_classes[0]])
                im2_file = random.choice(self.dataset[chosen_classes[1]])

            else:
                # Positive
                lb = torch.Tensor([1]).to(self.device).to(torch.long)
                chosen_class = random.choice(self.valid_classes)
                im1_file, im2_file = random.sample(self.dataset[chosen_class], k=2)

            im1 = load_image(im1_file, self.size, self.device)
            im2 = load_image(im2_file, self.size, self.device)

            return im1, im2, lb

class StaticDataset(Dataset):
    def __init__(self, protocol_file, size=[256, 256], partition='train',
                device='cpu', seed=None):
        with open(protocol_file, "r") as fd:
            self.protocol = json.load(fd)
        
        if partition == 'train':
            self.dataset = self.protocol['training_instances']
        elif partition == 'test_known'
            self.dataset = self.protocol['test_instances_known']
        elif partition == 'test_mixed'
            self.dataset = self.protocol['test_instances_mixed']
        elif partition == 'test_unique'
            self.dataset = self.protocol['test_instances_unique']

        self.use_triplet = self.protocol['config']['type'] == "triplet"
        self.device = device
        self.size = size

        if seed is None:
            seed = int(time.time())
        self.seed = seed
        # Seed for random is set on main, this is just for logging

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.use_triplet:
            anchor_file, pos_file, neg_file = self.dataset[idx]

            anchor = load_image(anchor_file, self.size, self.device)
            pos = load_image(pos_file, self.size, self.device)
            neg = load_image(neg_file, self.size, self.device)

            return anchor, pos, neg
        else:
            im1_file, im2_file, label = self.dataset[idx]

            lb = torch.Tensor([label]).to(self.device).to(torch.long)
            im1 = load_image(im1_file, self.size, self.device)
            im2 = load_image(im2_file, self.size, self.device)

            return im1, im2, lb
