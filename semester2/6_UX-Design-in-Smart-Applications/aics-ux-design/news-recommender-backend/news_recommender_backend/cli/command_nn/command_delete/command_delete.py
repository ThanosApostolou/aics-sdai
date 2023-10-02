from argparse import _SubParsersAction, ArgumentParser, Namespace, BooleanOptionalAction
import logging

from news_recommender_backend.cli.command_nn.command_delete.actions_delete import do_delete
from news_recommender_backend.domain.global_state.global_state import get_global_state



def handle_command_delete(args: Namespace):
    logging.debug("start handle_command_delete")

    global_state = get_global_state()
    errors = do_delete(global_state, args.name)
    if (len(errors) > 0):
        for error in errors:
            logging.error(f"ERROR: {error}")
    else:
        print('nn_model created successfully')
    logging.debug("end handle_command_delete")


def create_command_delete(command_root_subparsers: _SubParsersAction):
    command_delete_parser: ArgumentParser = command_root_subparsers.add_parser("delete")
    command_delete_parser.set_defaults(func=handle_command_delete)
    command_delete_parser.add_argument("--name", type=str, action="store", help="name of neural network", required=True)
