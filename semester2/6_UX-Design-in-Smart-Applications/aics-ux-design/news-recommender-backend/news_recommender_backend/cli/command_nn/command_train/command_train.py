from argparse import _SubParsersAction, ArgumentParser, Namespace, BooleanOptionalAction
import logging

from news_recommender_backend.cli.command_nn.command_train.actions_train import do_train
from news_recommender_backend.domain.global_state.global_state import get_global_state



def handle_command_train(args: Namespace):
    logging.debug("start handle_command_train")

    global_state = get_global_state()
    errors = do_train(global_state, args.name)
    if (len(errors) > 0):
        for error in errors:
            logging.error(f"ERROR: {error}")
    else:
        print('nn_model created successfully')
    logging.debug("end handle_command_train")


def create_command_train(command_root_subparsers: _SubParsersAction):
    command_train_parser: ArgumentParser = command_root_subparsers.add_parser("train")
    command_train_parser.set_defaults(func=handle_command_train)
    command_train_parser.add_argument("--name", type=str, action="store", help="name of neural network", required=True)
