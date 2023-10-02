from argparse import _SubParsersAction, ArgumentParser, Namespace
import logging

from news_recommender_backend.cli.command_nn.command_create.command_create import create_command_create
from news_recommender_backend.cli.command_nn.command_delete.command_delete import create_command_delete
from news_recommender_backend.cli.command_nn.command_train.command_train import create_command_train
from news_recommender_backend.cli.command_nn.command_update.command_update import create_command_update




def create_command_nn(command_root_subparsers: _SubParsersAction):
    command_nn_parser: ArgumentParser = command_root_subparsers.add_parser("nn")
    command_nn_subparsers = command_nn_parser.add_subparsers()
    create_command_create(command_nn_subparsers)
    create_command_update(command_nn_subparsers)
    create_command_delete(command_nn_subparsers)
    create_command_train(command_nn_subparsers)
