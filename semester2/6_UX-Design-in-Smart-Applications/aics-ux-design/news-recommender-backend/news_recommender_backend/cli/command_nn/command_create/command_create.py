from argparse import _SubParsersAction, ArgumentParser, Namespace, BooleanOptionalAction
import logging

from news_recommender_backend.cli.command_nn.command_create.actions_create import do_create
from news_recommender_backend.domain.global_state.global_state import get_global_state



def handle_command_create(args: Namespace):
    logging.debug("start handle_command_create")

    global_state = get_global_state()
    errors = do_create(global_state, args.name, args.active)
    if (len(errors) > 0):
        for error in errors:
            logging.error(f"ERROR: {error}")
    else:
        print('nn_model created successfully')
    logging.debug("end handle_command_create")


def create_command_create(command_root_subparsers: _SubParsersAction):
    command_create_parser: ArgumentParser = command_root_subparsers.add_parser("create")
    command_create_parser.set_defaults(func=handle_command_create)
    command_create_parser.add_argument("--name", type=str, action="store", help="name of neural network", required=True)
    command_create_parser.add_argument("--active", action=BooleanOptionalAction, help="if neural network will be active", required=True)
