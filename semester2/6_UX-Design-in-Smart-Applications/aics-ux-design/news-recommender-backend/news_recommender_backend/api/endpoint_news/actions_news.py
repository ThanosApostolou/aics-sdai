import logging
from fastapi import APIRouter, Request
from sqlalchemy.orm import Session
import tensorflow as tf

import news_recommender_backend.domain.auth.auth_service as auth_service
from news_recommender_backend.domain.auth.user_details import UserDetails
from news_recommender_backend.domain.entities.entities import User
from news_recommender_backend.domain.global_state.global_state import GlobalState
from news_recommender_backend.domain.news.news_constants import NewsCategoriesEnum
from news_recommender_backend.domain.nn import nn_service
from news_recommender_backend.domain.repositories import nn_model_repository, user_repository

def do_recommend_category(global_state: GlobalState, user_details: UserDetails) -> NewsCategoriesEnum:
    with Session(global_state.db_engine) as session:
        errors: list[str] = []
        user: User | None = user_repository.find_user_by_id(session, user_details.user_id)
        assert user is not None
        active_nn_model = nn_model_repository.find_active_nn_model(session)
        assert active_nn_model is not None
        assert active_nn_model.is_trained
        path = global_state.rootPath.joinpath(f"./server-data/{active_nn_model.name}.h5").resolve()
        mymodel = tf.keras.models.load_model(path, compile=True)
        assert mymodel is not None
        category: NewsCategoriesEnum = nn_service.predict_model_for_user(mymodel, user)


        return category