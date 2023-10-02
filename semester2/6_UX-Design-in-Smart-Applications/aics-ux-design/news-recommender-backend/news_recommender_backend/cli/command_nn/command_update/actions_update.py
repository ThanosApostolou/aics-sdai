import logging
import os
from sqlalchemy.orm import Session
import tensorflow as tf

from news_recommender_backend.domain.entities.entities import Base, NnModel
from news_recommender_backend.domain.global_state.global_state import GlobalState, get_global_state
from news_recommender_backend.domain.nn import nn_service
from news_recommender_backend.domain.repositories import nn_model_repository, user_repository


def do_update(global_state: GlobalState, name: str, active: bool) -> list[str]:
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

        if active:
            active_nn_model = nn_model_repository.find_active_nn_model(session)
            if active_nn_model is not None and active_nn_model.name != name:
                active_nn_model.is_active = False

        existing_model.is_active = active
        transaction.commit()
        return errors