from sqlalchemy import select, insert
from sqlalchemy.orm import Session

from news_recommender_backend.domain.entities.entities import User, UserQuestionnaire

def insert_questionnaire(session: Session, user_questionnaire: UserQuestionnaire) -> UserQuestionnaire | None:
    session.add(user_questionnaire)
    return user_questionnaire