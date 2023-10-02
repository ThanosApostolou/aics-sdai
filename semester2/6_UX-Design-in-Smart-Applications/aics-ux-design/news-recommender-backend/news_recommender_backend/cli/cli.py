from argparse import ArgumentParser

from news_recommender_backend.cli.command_root import create_command_root, handle_command_root

def create_cli():
    parser = ArgumentParser()
    create_command_root(parser)
    return parser
