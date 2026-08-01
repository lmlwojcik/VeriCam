from time import time
import json
import argparse

def gen_parser():
    parser = argparse.ArgumentParser()

    # Model / Experiment arguments
    parser.add_argument('-mc', '--model_config', default="configs/models/vit.json", type=str)
    parser.add_argument('-rt', '--run_type', default="inter", type=str)
    parser.add_argument('-n', '--run_name', default="test_drive_v0", type=str)
    parser.add_argument('-s', '--seed', default=None, type=int)
    parser.add_argument('-sp', '--save_path', default="results/", type=str)
    parser.add_argument('-r', '--resume', action='store_true')

    # Data arguments
    parser.add_argument('-dc', '--dataset_config', default=None, type=str)
    parser.add_argument('-i', '--images_dir', default="./datasets/protocol.json", type=str)
    parser.add_argument('-p', '--protocol', default="./datasets/protocol.json", type=str)
    parser.add_argument('-ca', '--cam_annotations', default="./datasets/protocol.json", type=str)

    # Training arguments
    parser.add_argument('-tc', '--train_config', default=None, type=str)
    parser.add_argument('-ec', '--test_config', default=None, type=str)
    parser.add_argument('-pc', '--predict_config', default=None, type=str)

    # Training parameters fine-tuners (overrides config files)
    # TO-DO: update these args
    parser.add_argument('-b', '--batch_size', default=None, type=int)
    parser.add_argument('-l', '--loss', default=None, type=str)
    parser.add_argument('-lr', '--learning_rate', default=None, type=str)
    parser.add_argument('-op', '--optimizer', default=None, type=str)
    parser.add_argument('-sc', '--scheduler', default=None, type=str)

    return parser

def get_args():
    parser = gen_parser()

    clargs = parser.parse_args()
    with open(clargs.model_config, "r") as fd:
        model_config = json.load(fd)
    if clargs.seed is None:
        seed = int(time())
    
    if clargs.dataset_config is not None:
        with open(clargs.dataset_config, 'r') as fd:
            data_args = json.load(fd)
    else:
        data_args = {
            'images_dir': clargs.images_dir,
            'protocol': clargs.protocol,
            'cam_annotations': clargs.cams_annotations
        }

    experiment_args = {
        'model_config': model_config,
        'run_name': clargs.run_name,
        'run_type': clargs.run_type,
        'resume': clargs.resume,
        'seed': seed,
        'data_args': data_args,
        'save_path': clargs.save_path
    }

    if clargs.train_config is not None:
        with open(clargs.train_config, "r") as fd:
            train_config = json.load(fd)
    else:
        train_config = None

    if clargs.test_config is not None:
        with open(clargs.test_config, "r") as fd:
            test_config = json.load(fd)
    else:
        test_config = None

    if clargs.predict_config is not None:
        with open(clargs.predict_config, "r") as fd:
            predict_config = json.load(fd)
    else:
        predict_config = None

    training_args = {
        'training_config': train_config,
        'test_config': test_config,
        'predict_config': predict_config
    }

    return experiment_args, training_args






