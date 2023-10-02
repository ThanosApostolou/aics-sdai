import logging
import os
from sqlalchemy.orm import Session

from news_recommender_backend.domain.entities.entities import Base, NnModel
from news_recommender_backend.domain.global_state.global_state import GlobalState, get_global_state
from news_recommender_backend.domain.nn import nn_service
from news_recommender_backend.domain.repositories import nn_model_repository


def do_create(global_state: GlobalState, name: str, active: bool) -> list[str]:
    logging.info(f"name={name}, active={active}")
    with Session(global_state.db_engine) as session, session.begin() as transaction:
        errors: list[str] = []
        if (len(name) == 0):
            errors.append('name shouldnt be empty')
            transaction.rollback()
            return errors

        existing_model_with_same_name = nn_model_repository.find_nn_model_by_name(session, name)
        if existing_model_with_same_name is not None:
            errors.append('there already is a ')
            transaction.rollback()
            return errors

        if active:
            active_nn_model = nn_model_repository.find_active_nn_model(session)
            if active_nn_model is not None and active_nn_model.name != name:
                active_nn_model.is_active = False

        model = nn_service.create_model()
        model.summary()
        os.makedirs(global_state.rootPath.joinpath("./server-data"), exist_ok=True)
        path = global_state.rootPath.joinpath(f"./server-data/{name}.h5").resolve()
        model.save(path, overwrite=True, save_format="h5")

        nn_model = NnModel(name=name, path=str(path), is_trained=False, is_active=active)
        nn_model_repository.insert_nn_model(session, nn_model)

        transaction.commit()
        return errors