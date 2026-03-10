import json
import argparse

def gen_parser():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('-c', '--model_config', default="configs/models/resnet50.json", type=str)
    parser.add_argument('-n', '--run_name', default="test_drive_v0", type=str)
    parser.add_argument('-r', '--resume', action='store_true')

    # Training arguments
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-tc', '--train_config', default="configs/train/trainer_baseline.json", type=str)
    parser.add_argument('-v', '--validate', action='store_true')
    parser.add_argument('-tv', '--validation_config', default="configs/train/validation_baseline.json", type=str)
    parser.add_argument('-p', '--predict', action='store_true') # test
    parser.add_argument('-tp', '--predict_config', default="configs/train/evaluation_baseline.json", type=str)

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

    experiment_args = {
        'model_config': model_config,
        'run_name': clargs.run_name,
        'resume': clargs.resume
    }

    if clargs.train:
        with open(clargs.train_config, "r") as fd:
            train_config = json.load(fd)
    else:
        train_config = None

    if clargs.validate:
        with open(clargs.validation_config, "r") as fd:
            validation_config = json.load(fd)
    else:
        validation_config = None

    if clargs.predict:
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






