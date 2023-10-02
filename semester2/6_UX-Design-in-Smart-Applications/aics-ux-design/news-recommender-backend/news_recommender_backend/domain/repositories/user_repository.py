from sqlalchemy import select, insert
from sqlalchemy.orm import Session

from news_recommender_backend.domain.entities.entities import User


def find_user_by_firebase_user_id(session: Session, firebase_user_id: str) -> User | None:
    stmt = select(User).where(User.firebase_user_id == firebase_user_id)
    result = session.scalars(stmt).one_or_none()
    return result


def find_user_by_id(session: Session, user_id: int) -> User | None:
    print('user_id', user_id)
    stmt = select(User).join(User.questionnaire, isouter=True).where(User.user_id == user_id)
    result = session.scalars(stmt).one_or_none()
    print('result', result)
    return result

def insert_user(session: Session, user: User) -> User:
    session.add(user)
    return user


def find_users_with_questionnaire(session: Session) -> list[User]:
    stmt = select(User).join(User.questionnaire, isouter=False).where(User.questionnaire != None)
    result = session.scalars(stmt).all()
    print('result', result)
    return list(result)