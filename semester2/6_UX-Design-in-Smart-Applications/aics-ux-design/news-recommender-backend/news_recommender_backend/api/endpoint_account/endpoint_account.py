
import logging
import traceback
from fastapi import APIRouter, HTTPException, Request
from news_recommender_backend.api.endpoint_account import actions_account
from news_recommender_backend.api.endpoint_account.dtos_account import AccountDetailsDto
import news_recommender_backend.domain.auth.auth_service as auth_service
from sqlalchemy.orm import Session

from news_recommender_backend.domain.auth.user_details_dto import UserDetailsDto
from news_recommender_backend.domain.global_state.global_state import GlobalState, get_global_state

def create_endpoint_account():
    endpoint_account = APIRouter(
        prefix="/account",
        tags=["account"],
        responses={404: {"description": "Not found"}},
    )


    @endpoint_account.get("/testAccountEndpoint")
    async def test_account_endpoint(request: Request) -> dict:
        globalState = get_global_state()
        with Session(globalState.db_engine) as session, session.begin() as transaction:
            # user_details = auth_service.authorize_user(request, session)

            return {"testAccount": "testAccount"}

    @endpoint_account.get("/fetchAccountDetails")
    async def handle_fetch_account_details(request: Request) -> AccountDetailsDto:
        try:
            globalState = get_global_state()
            user_details = auth_service.authorize_user(request)
            print('user_details', user_details)
            acount_details_dto: AccountDetailsDto = actions_account.do_fetch_account_details(globalState, user_details)
            return acount_details_dto
        except Exception as error:
            logging.error(error)
            traceback.print_exc()
            raise HTTPException(status_code=500)


    @endpoint_account.post("/saveAccountDetails")
    async def handle_save_account_details(request: Request, account_details: AccountDetailsDto) -> list[str]:
        logging.info(f"account_details: {account_details}")
        try:
            globalState = get_global_state()
            user_details = auth_service.authorize_user(request)
            print('user_details', user_details)
            errors = actions_account.do_save_account_details(globalState, user_details, account_details)
            return errors
        except Exception as error:
            logging.error(error)
            traceback.print_exc()
            raise HTTPException(status_code=500)


    return endpoint_account