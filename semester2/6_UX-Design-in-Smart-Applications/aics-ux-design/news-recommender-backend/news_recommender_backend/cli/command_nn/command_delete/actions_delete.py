import logging
import os
from sqlalchemy.orm import Session

from news_recommender_backend.domain.entities.entities import Base, NnModel
from news_recommender_backend.domain.global_state.global_state import GlobalState, get_global_state
from news_recommender_backend.domain.repositories import nn_model_repository


def do_delete(global_state: GlobalState, name: str) -> list[str]:
    logging.info(f"name={name}")
    with Session(global_state.db_engine) as session, session.begin() as transaction:
        errors: list[str] = []
        if (len(name) == 0):
            errors.append('name shouldnt be empty')
            transaction.rollback()
            return errors

        existing_model_with_same_name = nn_model_repository.find_nn_model_by_name(session, name)
        if existing_model_with_same_name is None:
            errors.append(f"there is no model with name={name}")
            transaction.rollback()
            return errors


        path = global_state.rootPath.joinpath(f"./server-data/{name}.h5").resolve()
        os.remove(path)

        nn_model_repository.delete_nn_model(session, existing_model_with_same_name)
        transaction.commit()
        return errors