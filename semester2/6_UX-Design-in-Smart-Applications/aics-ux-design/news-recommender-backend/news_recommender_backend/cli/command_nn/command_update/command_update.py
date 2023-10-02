from argparse import _SubParsersAction, ArgumentParser, Namespace, BooleanOptionalAction
import logging

from news_recommender_backend.cli.command_nn.command_update.actions_update import do_update
from news_recommender_backend.domain.global_state.global_state import get_global_state



def handle_command_update(args: Namespace):
    logging.debug("start handle_command_update")

    global_state = get_global_state()
    errors = do_update(global_state, args.name, args.active)
    if (len(errors) > 0):
        for error in errors:
            logging.error(f"ERROR: {error}")
    else:
        print('nn_model created successfully')
    logging.debug("end handle_command_update")


def create_command_update(command_root_subparsers: _SubParsersAction):
    command_update_parser: ArgumentParser = command_root_subparsers.add_parser("update")
    command_update_parser.set_defaults(func=handle_command_update)
    command_update_parser.add_argument("--name", type=str, action="store", help="name of neural network", required=True)
    command_update_parser.add_argument("--active", action=BooleanOptionalAction, help="if neural network will be active", required=True)
