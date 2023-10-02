
from fastapi import APIRouter, Request
from news_recommender_backend.api.endpoint_account.dtos_account import AccountDetailsDto, UserQuestionnaireDto
import news_recommender_backend.domain.auth.auth_service as auth_service
from sqlalchemy.orm import Session
from news_recommender_backend.domain.auth.user_details import UserDetails

from news_recommender_backend.domain.auth.user_details_dto import UserDetailsDto
from news_recommender_backend.domain.entities.entities import UserQuestionnaire
from news_recommender_backend.domain.global_state.global_state import GlobalState, get_global_state
from news_recommender_backend.domain.repositories import user_questionnaire_repository, user_repository

def do_fetch_account_details(globalState: GlobalState, user_details: UserDetails) -> AccountDetailsDto:
    with Session(globalState.db_engine) as session:
        user = user_repository.find_user_by_id(session, user_details.user_id)
        assert user is not None
        user_questionnaire = None if user.questionnaire is None else UserQuestionnaireDto(favorite_category=user.questionnaire.favorite_category, is_extrovert=user.questionnaire.is_extrovert, age=user.questionnaire.age, educational_level=user.questionnaire.educational_level, sports_interest=user.questionnaire.sports_interest, arts_interest=user.questionnaire.arts_interest)
        account_details_dto = AccountDetailsDto(name=user.name, email=user.email, user_questionnaire=user_questionnaire)
        return account_details_dto


def do_save_account_details(globalState: GlobalState, user_details: UserDetails, account_details: AccountDetailsDto) -> list[str]:
    with Session(globalState.db_engine) as session, session.begin() as transaction:
        errors: list[str] = []
        user = user_repository.find_user_by_id(session, user_details.user_id)
        assert user is not None
        existing_questionnaire = user.questionnaire
        if account_details.user_questionnaire is None:
            existing_questionnaire = None
        else:
            if existing_questionnaire is None:
                existing_questionnaire = UserQuestionnaire(user_id_fk=user.user_id, user = user)
            existing_questionnaire.favorite_category = account_details.user_questionnaire.favorite_category
            existing_questionnaire.is_extrovert = account_details.user_questionnaire.is_extrovert
            existing_questionnaire.age = account_details.user_questionnaire.age
            existing_questionnaire.educational_level = account_details.user_questionnaire.educational_level
            existing_questionnaire.sports_interest = account_details.user_questionnaire.sports_interest
            existing_questionnaire.arts_interest = account_details.user_questionnaire.arts_interest

            user_questionnaire_repository.insert_questionnaire(session, existing_questionnaire)


        user.questionnaire = existing_questionnaire
        user_repository.insert_user(session, user)

        if not errors :
            session.commit()
        else:
            session.rollback()

        return errors
