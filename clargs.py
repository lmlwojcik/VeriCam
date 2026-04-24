from time import time
import json
import argparse

def gen_parser():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('-c', '--model_config', default="configs/models/vit.json", type=str)
    parser.add_argument('-n', '--run_name', default="test_drive_v0", type=str)
    parser.add_argument('-s', '--seed', default=None, type=int)
    parser.add_argument('-r', '--resume', action='store_true')

    # Data arguments
    parser.add_argument('-d', '--dataset' default="./datasets/protocol.json", type=str)

    # Training arguments
    parser.add_argument('-t', '--train_config', default=None, type=str)
    parser.add_argument('-v', '--test_config', default=None, type=str)
    parser.add_argument('-p', '--predict_config', default=None, type=str)

    # Training parameters fine-tuners (overrides config files)
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

    experiment_args = {
        'model_config': model_config,
        'run_name': clargs.run_name,
        'seed': seed,
        'resume': clargs.resume
    }

    if clargs.train_config is not None:
        with open(clargs.train_config, "r") as fd:
            train_config = json.load(fd)
    else:
        train_config = None

    if clargs.test_config is not None:
        with open(clargs.validation_config, "r") as fd:
            validation_config = json.load(fd)
    else:
        validation_config = None

    if clargs.predict_config is not None:
        with open(clargs.predict_config, "r") as fd:
            predict_config = json.load(fd)
    else:
        predict_config = None

    training_args = {
        'training_config': train_config,
        'validation_config': validation_config,
        'predict_config': predict_config
    }

    return experiment_args, training_args






