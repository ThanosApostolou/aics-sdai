from argparse import _SubParsersAction, ArgumentParser, Namespace
import logging

from news_recommender_backend.cli.command_db.actions_db import do_init_db



def handle_command_db(args: Namespace):
    logging.debug("start handle_command_db")
    print('args\n', args)
    if args.action == "init":
        do_init_db()
    logging.debug("end handle_command_db")


def create_command_db(command_root_subparsers: _SubParsersAction):
    command_db_parser = command_root_subparsers.add_parser("db")
    command_db_parser.set_defaults(func=handle_command_db)
    command_db_parser.add_argument('action', type=str, choices=["init"])
