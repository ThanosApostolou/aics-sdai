from argparse import Namespace
import logging
import os


from news_recommender_backend.api.api import create_api
from news_recommender_backend.cli.cli import create_cli
from news_recommender_backend.domain.core.db import create_db_engine
from news_recommender_backend.domain.core.settings import Settings, create_settings
from news_recommender_backend.domain.firebase.firebase_service import create_firebase_app
from news_recommender_backend.domain.global_state.global_state import initialize_global_state


def main_cli():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("start main")
    settings = create_settings()
    db_engine = create_db_engine(settings)
    firebaseApp = create_firebase_app()
    initialize_global_state(None, settings, db_engine, firebaseApp)
    parser = create_cli()
    args: Namespace = parser.parse_args()
    args.func(args)
    logging.info("end main")


if __name__ == "__main__":
    main_cli()