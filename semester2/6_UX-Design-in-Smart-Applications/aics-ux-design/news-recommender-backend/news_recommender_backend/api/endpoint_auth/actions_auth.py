import logging
from fastapi import Request
from sqlalchemy.orm import Session

from news_recommender_backend.domain.auth import auth_service
from news_recommender_backend.domain.auth.firebase_user import FirebaseUser

from news_recommender_backend.domain.auth.user_details_dto import UserDetailsDto
from news_recommender_backend.domain.entities.entities import User
from news_recommender_backend.domain.global_state.global_state import GlobalState
from news_recommender_backend.domain.repositories import user_repository


async def do_fetch_user_details(globalState: GlobalState, firebase_user: FirebaseUser) -> UserDetailsDto:
    with Session(globalState.db_engine) as session, session.begin() as transaction:
        user = user_repository.find_user_by_firebase_user_id(session, firebase_user.firebase_user_id)
        if user is None:
            user = User(firebase_user_id=firebase_user.firebase_user_id, email=firebase_user.email, name=firebase_user.name)
            user_repository.insert_user(session, user)
            session.flush()
        else:
            user.email = firebase_user.email
            user.name = firebase_user.name

        user_details = auth_service.get_user_details(firebase_user, user)
        logging.debug(f"user_details: {user_details}")
        userDetailsDto = UserDetailsDto.fromUserDetails(user_details)
        session.commit()
        return userDetailsDto