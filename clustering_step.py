import numpy as np
import torch
from torch.nn.functional import cosine_similarity, normalize
from tqdm import tqdm
from glob import glob
import argparse
from natsort import natsorted

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
import igraph as ig
import leidenalg as la

import os
import json
import argparse

from sklearn.metrics import v_measure_score, adjusted_rand_score, homogeneity_completeness_v_measure

def gen_idxs(fs):
    st = sorted(fs)
    idxs = {x: i for i,x in enumerate(st)}
    return idxs, {x: i for i,x in idxs.items()}

def proc_fname(fname):
    return os.path.splitext(os.path.basename(fname))[0]

def build_initial_relations(fs, fts, gt, baseline, atoi):
    known = baseline['train']
    fs_gt = {proc_fname(x): int(k) for k,v in gt.items() for x in v if k not in ['null', 'unknown']}
    cam_idx = {}

    fmap = []
    fmap_idx = {}

    c_idx = 0
    for f in [proc_fname(x) for x in fs]:
        if f not in known:
            continue

        if fs_gt[f] not in cam_idx.keys():
            cam_idx[fs_gt[f]] = c_idx
            fmap_idx[cam_idx[fs_gt[f]]] = []
            c_idx += 1

        fmap_idx[cam_idx[fs_gt[f]]].append(atoi[f])
        fmap.append(cam_idx[fs_gt[f]])

    print(len(set(fmap)), len(cam_idx.values()))

    return fmap, fmap_idx, cam_idx, fs_gt


def compare(t1, t2, thresh=None):
    score = cosine_similarity(t1.squeeze(), t2.squeeze(), dim=0)
    if thresh is not None:
        return True if score >= thresh else False
    return score

def vector_compare(t1, t2):
    scores = cosine_similarity(t1.unsqueeze(0), t2, dim=1)
    return scores

# def matrix_compare(t1, t2):
#     scores = cosine_similarity(t1, t2, dim=1)
#     return scores

def matrix_compare(t1, t2):
    t1 = t1.squeeze(1)
    t2 = t2.squeeze(1)
    t1_norm = t1 / t1.norm(dim=1)[:, None]
    t2_norm = t2 / t2.norm(dim=1)[:, None]
    scores = torch.mm(t1_norm, t2_norm.transpose(0,1))
    return scores

def get_distance(t1, t2):
    score = cosine_similarity(t1.squeeze(), t2.squeeze(), dim=0)
    dist = 1 - score
    return dist

def vector_distance(t1, t2):
    scores = cosine_similarity(t1.unsqueeze(0), t2, dim=1)
    dists = 1 - scores
    return dists

def create_matrices(fdir):
    fs = [proc_fname(x) for x in glob(fdir + "*")]
    ftoi, itof = gen_idxs(fs)
    tensors = []
    for k,v in tqdm(ftoi.items()):
        tensors.append(torch.load(fdir + k + ".pt"))
    tensors = torch.stack(tensors)
    matrix = matrix_compare(tensors, tensors)
    return tensors, matrix, ftoi

def proc_sims(sims, valid_idxs):
    fts = sims.detach()
    fts = fts[np.ix_(valid_idxs, valid_idxs)]
    #fts[fts >= 0.5] = 1
    fts = 0.5 * (fts + fts.transpose(-2, -1))
    fts[fts <= 0.5] = 0
    fts.fill_diagonal_(1)
    fts = fts.cpu().numpy()
    return fts


def graph_naive(baseline, cam_gt, fmap_idx, relations, atoi, cam_idx):
    idx_cam = {v:[k] for k,v in cam_idx.items()}
    relations = relations.cpu()
    itoa = {v:k for k,v in atoi.items()}

    train = [proc_fname(x) for k in baseline['train'].values() for x in k]
    test = [proc_fname(x) for k in baseline['test'].values() for x in k]
    nfs = natsorted([proc_fname(x) for x in test])

    clust_idxs = list(fmap_idx.keys())
    clust_idxs = []
    print(fmap_idx)
    fmap_idx = {0: [atoi[nfs[0]]]}
    results = {}
    break_at = -1
    c = 0
    right = 0
    total = 0

    with tqdm(nfs[1:], total=len(nfs), dynamic_ncols=True) as step:
        for f in step:
            clust_idxs = list(fmap_idx.keys())

            scores = [torch.mean(relations[atoi[f],[fmap_idx[x]]]) for x in clust_idxs]
            v, idxs = torch.topk(torch.Tensor(scores), k=1)
            
            chosen_k = -1 if v[0] < 0.6 else clust_idxs[idxs[0]]

            if chosen_k != -1:
                # print(f"{f} assigned to cluster {chosen_k} ({idx_cam[chosen_k]}) (gt={cam_gt[f]}) with certainty={v[0]}")
                # print(f"gt camera known: {cam_gt[f] in cam_idx.keys()}")

                if cam_gt[f] in cam_idx.keys() and \
                        ((type(idx_cam[chosen_k]) == int and cam_gt[f] == idx_cam[chosen_k]) or \
                         (type(idx_cam[chosen_k]) == list and cam_gt[f] in idx_cam[chosen_k])):
                    right += 1

                fmap_idx[chosen_k].append(atoi[f])
                results[atoi[f]] = {'gt': cam_gt[f], 'pd': chosen_k}
            else:
                # print(f"{f} assigned to a new camera ({max(fmap_idx.keys()) + 1}) (gt={cam_gt[f]}).")
                # print(f"gt camera known: {cam_gt[f] in cam_idx.keys()}")
                if cam_gt[f] not in cam_idx.keys():
                    right += 1

                chosen_k = max(fmap_idx.keys()) + 1
                cam_idx[cam_gt[f]] = chosen_k
                if chosen_k not in idx_cam.keys():
                    idx_cam[chosen_k] = []
                idx_cam[chosen_k].append(cam_gt[f])
                fmap_idx[chosen_k] = [atoi[f]]
                results[atoi[f]] = {'gt': cam_gt[f], 'pd': chosen_k}

            gts, pds = zip(*[(x['gt'], x['pd']) for x in results.values()])
            c += 1
            if c == break_at:
                exit()

            total += 1
    return pds


def main(features, baseline, cams_file, use_known, output, method):
    sims = torch.load(f"{features}/similarity.pt")
    with open(f"{features}/idxs.json", "r") as fd:
        idxs = json.load(fd)

    with open(baseline, "r") as fd:
        baseline_known = json.load(fd)
    with open(cams_file, "r") as fd:
        cams = json.load(fd)

    baseline = {x: cams[str(x)] for x in baseline_known['test']}
    baseline_known = {x: cams[str(x)] for x in baseline_known['train']}

    valid_idxs = []
    known_partitions = []

    for c in baseline.keys():
        valid_idxs += [idxs[proc_fname(x)] for x in baseline[c]]
        known_partitions += [-1 for x in baseline[c]]
    
    if use_known:
        for c in baseline_known.keys():
            valid_idxs += [idxs[proc_fname(x)] for x in baseline_known[c]]
            known_partitions += [c for x in baseline_known[c]]
    test_idxs = [x for x,_ in enumerate(valid_idxs)]

    for i, val in enumerate(known_partitions):
        known_partitions[i] = int(known_partitions[i])+1

    fts = proc_sims(sims, valid_idxs)

    if method == 'leiden':
        G = ig.Graph.Adjacency((fts > 0).tolist(), mode="undirected")
        G = ig.Graph.Weighted_Adjacency(fts, mode="undirected", attr="weight")

        partition = la.find_partition(
            G,
            la.CPMVertexPartition,
            initial_membership=known_partitions,
            weights='weight', # or None if weight 0 edges are absent from the graph
            resolution_parameter=0.8,
            seed=42,
            max_comm_size=50
        )
        # partition = spectral_clustering_arpack(fts, 122)
        ret = []
        for _, community in enumerate(partition):
            ret.append(community)
    
    elif method == 'naive':
        fmap_idx, cam_idx, fs_gt = build_initial_relations(cams, baseline, idxs)
        ret = graph_naive(baseline, fs_gt, fmap_idx, fts, idxs, cam_idx)


    with open(output, "w") as fd:
        json.dump(ret, fd, indent=2)

    valid_idxs = []

    for i, c in enumerate(baseline.values()):
        valid_idxs += [i for x in c]
    if use_known:
        for i, c in enumerate(baseline_known.values()):
            valid_idxs += [i for x in c]

    n = len(valid_idxs)
    print(n)

    comms = ret
    pred = [[] for x in range(n)]
    for k,v in enumerate(comms):
        for p in v:
            if p in test_idxs:
                pred[p] = k
    print(homogeneity_completeness_v_measure(valid_idxs,pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--features', type=str)
    parser.add_argument('-b', '--baseline', type=str)
    parser.add_argument('-c', '--cams_file', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-m', '--method', type=str)

    args = vars(parser.parse_args())
    main(**args)

