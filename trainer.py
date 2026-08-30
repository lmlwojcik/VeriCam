import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from pathlib import Path
import datetime
import shutil
from glob import glob

from evaler import calc_metrics
from dataset import DynamicDataset, StaticDataset, StaticDatasetEval, MixedDatasetTrain
from utils import start_log, end_log, log_metrics_json, dict_to_table, log_config
from callbacks import start_training, update_step, conclude_slice

def cosine_distance(t1, t2):
    return 1 - cosine_similarity(t1, t2)

def training_loop_triplet(model, data, opt, loss, yield_cfg):

    training_data = start_training(yield_cfg)
    epoch = 0
    c_step = 0

    while epoch < yield_cfg['max_epochs']:
        model.train()
        s_loss = 0

        with tqdm(data, unit="batch", dynamic_ncols=True) as tepoch:            
            for sample in tepoch:
                tepoch.set_description(f"Epoch {int(epoch)}")
                c_step += 1

                anc, pos, neg = sample
                opt.zero_grad()

                logits_anc = model(anc)
                logits_pos = model(pos)
                logits_neg = model(neg)

                c_loss = loss(logits_anc, logits_pos, logits_neg)

                c_loss.backward()
                opt.step()
                s_loss += c_loss.item()

                training_data, reset = update_step(training_data, c_loss, False)
                
                if reset:
                    # Gather loss, update metrics and advance epoch
                    avg_loss = s_loss/c_step
                    c_step = 0
                    s_loss = 0
                    #train_metrics = gen_metrics(gts, pds, pt="train",
                    #                            cls=dataset['class_names'], loss=avg_loss)
                    training_data = conclude_slice(training_data)
                    #yield avg_loss, train_metrics
                    yield avg_loss, training_data

                tepoch.set_postfix(loss=c_loss.item())

        training_data, reset = update_step(training_data, c_loss, True)
        if reset:
            # Gather loss, update metrics and advance epoch
            avg_loss = s_loss/len(data)
            s_loss = 0
            #train_metrics = gen_metrics(gts, pds, pt="train",
            #                            cls=dataset['class_names'], loss=avg_loss)
            training_data = conclude_slice(training_data)
            #yield avg_loss, train_metrics
            yield avg_loss, training_data
        epoch += 1
    yield 0.0, training_data


def train_model(model, cfg, data_args, log_cfg=None, seed=None, run_type='intra'):
    print(cfg, data_args, model)

    # Experiment setup
    save_path = Path(cfg['save_path'])
    save_path.mkdir(parents=True,exist_ok=True)

    if log_cfg is not None:
        log_file = save_path / Path(log_cfg['experiment_name'] + ".json")
        start_log(log_file)
    log_metrics = {}
    best_metrics = {}

    if cfg['use_gpu'] != -1:
        if len(cfg['use_gpu']) == 1:
            cfg['use_gpu'] = f"cuda:{cfg['use_gpu']}"
        model.to(torch.device(f"{cfg['use_gpu']}"))
    device = cfg['use_gpu']
    print("Running experiments at: ", device)

    # Generate training dataset and dataloader
    if run_type == 'inter':
        train_dataset = DynamicDataset(
            **data_args,
            n_samples=cfg['train_steps'],
            partition='train',
            device=device,
            seed=seed
        )
    else:
        train_dataset = MixedDatasetTrain(
            **data_args,
            n_samples=cfg['train_steps'],
            partition='train',
            device=device,
            seed=seed
        )

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'])

    # Generate validation dataset and dataloader
    if run_type == 'inter':
        val_dataset = DynamicDataset(
            **data_args,
            n_samples=cfg['val_steps'],
            partition='val',
            use_triplet=False,
            device=device,
            seed=seed
        )
    else:
        val_dataset = MixedDatasetTrain(
            **data_args,
            n_samples=cfg['val_steps'],
            partition='val',
            use_triplet=False,
            device=device,
            seed=seed
        )
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'])
    # if 'val_steps' not in cfg.keys():
    #     val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)
    #     split_val = False
    # else:
    #     n_loaders = len(val_dataset)//cfg['val_steps']
    #     val_datasets = random_split(val_dataset, [1/n_loaders]*n_loaders)
    #     val_loaders = []
    #     for vd in val_datasets:
    #         val_loaders.append(DataLoader(vd, batch_size=cfg['batch_size'], shuffle=False))
    #     c_val = 0
    #     split_val = True

    # Declaring loss, optimizer and scheduler if any
    if cfg['optim'] == "adam":
        opt = Adam(model.named_parameters(), **cfg['optim_config'])
    elif cfg['optim'] == "sgd":
        opt = SGD(model.named_parameters(), **cfg['optim_config'])
    scheduler = ReduceLROnPlateau(opt, 'min', **cfg['scheduler_config'])

    #loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.5)

    best_metric = 1e9 if cfg['es_metric'].endswith("loss") else 0
    start = datetime.datetime.now()
    yield_cfg = cfg['yield_config']
    cnt = 0

    # Training loop
    for avg_loss, training_data in training_loop_triplet(
        model,
        train_loader,
        opt,
        loss,
        yield_cfg,
    ):

        log_metrics['epoch'] = training_data['epoch']
        if log_cfg['method'] != 'epoch':
            log_metrics['step'] = training_data['step']
        log_metrics['train_loss'] = avg_loss

        if cfg['validate']: 
            vm = calc_metrics(model, val_loader, pt='val',
                            loss=None, device=cfg['use_gpu'])
            log_metrics.update(vm)
            scheduler.step(log_metrics['train_loss'])
            log_metrics['lr'] = scheduler.get_last_lr()[-1]

            #     vm = calc_metrics(model, val_loaders[c_val], pt='val',
            #                     loss=None, device=cfg['use_gpu'])
            #     log_metrics.update(vm)
            #     scheduler.step(log_metrics['train_loss'])
            #     log_metrics['lr'] = scheduler.get_last_lr()[-1]
            #     c_val = (c_val+1)%len(val_loaders)

        log_metrics['minutes'] = (datetime.datetime.now() - start).total_seconds()/60

        print(dict_to_table(log_metrics))
        if log_cfg is not None:
            log_metrics_json(log_metrics, log_file)

        if cfg['do_es']:
            cnt += 1
            current_metric = log_metrics[cfg['es_metric']]
            if (cfg['es_metric'].endswith("loss") and current_metric < best_metric) \
                    or (current_metric > best_metric):
                print("Saving best model at Epoch ", training_data['epoch'])
                best_metric = current_metric
                best_metrics.update(log_metrics)
                cnt = 0
                
                if cfg['save_best']:
                    torch.save(model.state_dict(), save_path / Path("model_best.pth"))
            if cnt >= cfg['patience']:
                break

    epoch = training_data['epoch']
    if cfg['save_last']:
        torch.save(model.state_dict(), save_path / Path(f"model_last_epoch_{int(epoch)}.pth"))

    if epoch == 0:
        log_metrics = calc_metrics(model, train_loader, pt='train',
                                   loss=None, device=cfg['use_gpu'])
        vm = calc_metrics(model, val_loader, pt='val',
                          loss=None, device=cfg['use_gpu'])
        log_metrics.update(vm)
    else:
        model.load_state_dict(torch.load(save_path / Path(f"model_best.pth"), weights_only=True))
    if log_cfg is not None:
        log_metrics_json(log_metrics, log_file)
        end_log(log_file)
    cfg['save_path'] = str(cfg['save_path'])
    log_config(save_path, {"cfg": cfg, "data_args": data_args})

    return model, best_metrics

def eval_model(model, cfg, dataset, log_cfg=None, seed=None, run_type='inter'):
    print(cfg, dataset, model)

    if cfg['use_gpu'] != -1:
        if len(cfg['use_gpu']) == 1:
            cfg['use_gpu'] = f"cuda:{cfg['use_gpu']}"
        model.to(torch.device(f"{cfg['use_gpu']}"))
    device = cfg['use_gpu']
    print("Running experiments at", cfg['use_gpu'])

    # Generate validation dataset and dataloader
    eval_dataset = StaticDatasetEval(**dataset, n_samples=-1, device=device, seed=seed)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg['batch_size'], shuffle=False)
    metrics_eval = calc_metrics(model, eval_loader, pt='test_known')

    return metrics_eval


def predict_model(model, cfg, dataset, log_cfg=None):
    pass
