import torch

def start_training(cfg, max_steps=None):

    training_data = {
        'epoch': 1,
        'step': 0,
        'global_step': 0,
        's_loss': 0,

        'method': cfg['method'],
        'max_steps': cfg['max_steps'],
        #'train_gts': torch.tensor([]).to(device),
        #'train_pds': torch.tensor([]).to(device),
        'logs': {}
    }

    training_data['logs']['checkpoint'] = []
    for k in cfg['logs']:
        training_data['logs'][k] = []

    return training_data

def update_step(training_data, s_loss, epoch_end):
    training_data['step'] += 1
    training_data['global_step'] += 1

    if epoch_end:
        training_data['step'] = 0
        training_data['epoch'] += 1

        if training_data['method'] == 'epoch':
            return training_data, True
    
    elif training_data['method'] == 'steps' \
            and training_data['global_step'] % training_data['max_steps'] == 0:
        return training_data, True

    return training_data, False

def conclude_slice(training_data):

    # training_data['train_gts'] = torch.tensor([]).to(device)
    # training_data['train_pds'] = torch.tensor([]).to(device)

    return training_data

def log_slice(cfg, training_data, loggables):
    epoch = training_data['epoch']
    step = training_data['step']

    checkpoint = epoch if cfg['method'] == 'epoch' else f"{epoch}_{step}"
    training_data['logs']['checkpoint'].append(checkpoint)

    for k,v in loggables.items():
        training_data['logs'][k].append(v)

    return training_data
