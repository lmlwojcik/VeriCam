from clargs import get_args
from models import create_model, get_model_with_weights

def main(e_args, t_args):

    model = create_model(e_args['model_config'])

    if t_args['training_config'] is not None:
        # train

    if t_args['predict_config'] is not None:
        # predict


if __name__ == "__main__":
    experiment_args, training_args = get_args()
    main(experiment_args, training_args)