import logging
from argparse import ArgumentParser, Namespace

from news_recommender_backend.cli.command_db.command_db import create_command_db, handle_command_db
from news_recommender_backend.cli.command_nn.command_nn import create_command_nn


def handle_command_root(args: Namespace):
    logging.debug("start handle_command_root")
    print('args\n', args)
    logging.debug("end handle_command_root")


def create_command_root(parser: ArgumentParser):
    parser.set_defaults(func=handle_command_root)
    command_root_subparsers = parser.add_subparsers()
    create_command_db(command_root_subparsers)
    create_command_nn(command_root_subparsers)
