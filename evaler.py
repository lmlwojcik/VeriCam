import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torcheval.metrics.functional import (
        binary_accuracy,
        binary_f1_score,
        binary_confusion_matrix
    )

def gen_metrics(model, dl, loss = None, device = None, verbose = True):
    pds = []
    gts = []

    if loss is not None:
        vloss = 0
        idx = 0
    else:
        vloss = None

    if verbose:
        dl = tqdm(dl, dynamic_ncols=True)
        
    model.eval()
    with torch.no_grad():
        for sample in dl:
            im1, im2, lb = sample
            
            ft1 = model(im1)
            ft2 = model(im2)

            for g, p, l in zip(ft1, ft2, lb):
                sim = cosine_similarity(g.squeeze(), p.squeeze(), dim=0)
                pd = 1 if sim > 0.5 else 0

                pds.append(pd)
                gts.append(l[0].item())
            # if loss is not None:
            #     vloss += loss(logits, lb).item()
            #     idx += 1

    gts = torch.Tensor(gts).to(device)
    pds = torch.Tensor(pds).to(device)

    if loss is not None:
        vloss /= idx

    return gts, pds, vloss

def calc_metrics(model, dl, pt='train', loss=None, device=None, return_matrix=False):
    gts, pds, vloss = gen_metrics(model, dl, loss, device)
    gts = gts.to(torch.int64)
    pds = pds.to(torch.int64)

    f1 = binary_f1_score(pds,gts).item()
    acc = binary_accuracy(pds,gts).item()

    metrics = {f"{pt}_acc": acc, f"{pt}_f1": f1}
    if loss is not None:
        metrics[f"{pt}_loss"] = vloss

    if return_matrix:
        cm = binary_confusion_matrix(pds,gts).tolist()
        metrics[f"{pt}_matrix"] = cm

        df = pd.DataFrame(cm, index=[False, True],
                              columns=[False,True]).rename_axis('Ground Truth',
                          axis='index').rename_axis('Prediction', axis='columns')
        plt.rcParams.update({'font.size': 15})
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
        ax = sns.heatmap(df, cmap=cmap, annot=True, fmt="d", linewidths=1, square=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return metrics

