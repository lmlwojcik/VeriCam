import random
import json
from pathlib import Path

from clargs import get_args
from trainer import train_model, predict_model, eval_model
from models import create_model, get_model_with_weights

def main(e_args, t_args):
    dataset = e_args['data_args']
    run_name = e_args['run_name']
    save_path = Path(e_args['save_path'])

    if run_name is not None:
        print(f"Starting run: {run_name}")
        save_path = save_path / Path(run_name)
    else:
        print("Starting run")

    #print(f"Experiment configuration: ", e_args['training_config'])

    random.seed(e_args['seed'])
    model = create_model(e_args['model_config'])

    if t_args['training_config'] is not None:
        # Train model
        cfg = t_args['training_config']
        cfg['save_path'] = save_path
        model, metrics = train_model(model, cfg, e_args['data_args'], log_cfg=cfg['log_config'], run_type=e_args['run_type'])

    if t_args['test_config'] is not None:
        # Eval
        if t_args['training_config'] is None: # If model wasn't trained...
            model = get_model_with_weights(model, None, save_path, 'cuda:0') # Load from checkpoint.

        cfg = t_args['test_config']
        cfg['save_path'] = save_path
        metrics = eval_model(model, cfg, e_args['data_args'], log_cfg=cfg['log_config'], run_type=e_args['run_type'])

        print(metrics)
        with open(Path(cfg['save_path']) / f"eval_results.json", "w") as fd:
            json.dump(metrics, fd, indent=2)

    if t_args['predict_config'] is not None:
        # predict
        cfg = t_args['predict_config']
        cfg['save_path'] = save_path
        model, metrics = predict_model(model, cfg, e_args['data_args'], log_cfg=cfg['log_config'], run_type=e_args['run_type'])


if __name__ == "__main__":
    experiment_args, training_args = get_args()
    main(experiment_args, training_args)