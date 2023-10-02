from sqlalchemy import select, insert
from sqlalchemy.orm import Session

from news_recommender_backend.domain.entities.entities import NnModel


def find_nn_model_by_name(session: Session, name: str) -> NnModel | None:
    stmt = select(NnModel).where(NnModel.name == name)
    result = session.execute(stmt).one_or_none()
    print('result', result)
    return None if result is None else result.tuple()[0]


def find_active_nn_model(session: Session) -> NnModel | None:
    stmt = select(NnModel).where(NnModel.is_active == True)
    result = session.execute(stmt).one_or_none()
    print('result', result)
    return None if result is None else result.tuple()[0]


def insert_nn_model(session: Session, nn_model: NnModel) -> NnModel:
    session.add(nn_model)
    return nn_model


def delete_nn_model(session: Session, nn_model: NnModel) -> NnModel:
    session.delete(nn_model)
    return nn_model