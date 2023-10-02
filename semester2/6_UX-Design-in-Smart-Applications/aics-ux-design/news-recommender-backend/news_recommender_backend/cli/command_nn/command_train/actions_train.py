import logging
import os
from sqlalchemy.orm import Session
import tensorflow as tf

from news_recommender_backend.domain.entities.entities import Base, NnModel
from news_recommender_backend.domain.global_state.global_state import GlobalState, get_global_state
from news_recommender_backend.domain.nn import nn_service
from news_recommender_backend.domain.repositories import nn_model_repository, user_repository


def do_train(global_state: GlobalState, name: str) -> list[str]:
    logging.info(f"name={name}")
    with Session(global_state.db_engine) as session, session.begin() as transaction:
        errors: list[str] = []
        if (len(name) == 0):
            errors.append('name shouldnt be empty')
            transaction.rollback()
            return errors

        existing_model = nn_model_repository.find_nn_model_by_name(session, name)
        if existing_model is None:
            errors.append(f"couldn't find nn model with name={name}")
            transaction.rollback()
            return errors

        path = global_state.rootPath.joinpath(f"./server-data/{name}.h5").resolve()
        model = tf.keras.models.load_model(path, compile=True)
        assert model is not None

        users = user_repository.find_users_with_questionnaire(session)
        for user in users:
            logging.info(f"user={user}")
        plot_file = global_state.rootPath.joinpath(f"./server-data/{name}-plot.png").resolve()
        model = nn_service.train_evaluate_model(model, users, plot_file)


        model.save(path, overwrite=True, save_format="h5")
        existing_model.is_trained = True

        transaction.commit()
        return errors