from fastapi import Depends, Request
from fastapi.security import HTTPBearer, OAuth2AuthorizationCodeBearer
from firebase_admin import auth
from sqlalchemy.orm import Session

from news_recommender_backend.domain.auth.firebase_user import FirebaseUser
from news_recommender_backend.domain.auth.user_details import UserDetails
from news_recommender_backend.domain.entities.entities import User
from news_recommender_backend.domain.global_state.global_state import get_global_state
from news_recommender_backend.domain.news.news_constants import NewsCategoriesEnum
from news_recommender_backend.domain.repositories import user_repository
# from firebase_admin import app_check

httpBearer = HTTPBearer()

async def firebase_authentication(token: str = Depends(httpBearer)):
    decoded_token = auth.verify_id_token(token)
    return decoded_token

# def create_oauth2():
#     oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
#     return oauth2_scheme

def authenticate_user(request: Request) -> FirebaseUser:
    headers = request.headers
    jwt = headers.get('Authorization')
    firebase_user_dict: dict = auth.verify_id_token(jwt)
    # firebaseUser = app_check.verify_token(jwt)
    print('firebaseUser\n', firebase_user_dict)
    firebase_user_id = firebase_user_dict["user_id"]
    email = firebase_user_dict["email"]
    name = firebase_user_dict["name"] if "name" in firebase_user_dict else ""
    assert firebase_user_id is not None
    assert email is not None
    assert name is not None

    uid: str = firebase_user_dict["uid"]
    assert uid is not None
    user_record = auth.get_user(uid)
    assert user_record is not None

    name = firebase_user_dict["name"] if "name" in firebase_user_dict else user_record.display_name
    assert firebase_user_id is not None

    firebase_user = FirebaseUser(firebase_user_id, email, name)
    return firebase_user


def get_user_details(firebase_user: FirebaseUser, user: User) -> UserDetails:
    favorite_category: NewsCategoriesEnum | None = None if user.questionnaire is None else user.questionnaire.favorite_category
    user_details = UserDetails(user_id=user.user_id, firebase_user_id=firebase_user.firebase_user_id, email=firebase_user.email, name=firebase_user.name, favorite_category=favorite_category)
    return user_details

def authorize_user(request: Request) -> UserDetails:
    with Session(get_global_state().db_engine) as session:
        firebase_user = authenticate_user(request)
        user = user_repository.find_user_by_firebase_user_id(session, firebase_user.firebase_user_id)
        assert user is not None
        return get_user_details(firebase_user, user)