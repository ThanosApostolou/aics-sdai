
import logging
from fastapi import APIRouter, Request
from news_recommender_backend.api.endpoint_auth import actions_auth
import news_recommender_backend.domain.auth.auth_service as auth_service

from news_recommender_backend.domain.auth.user_details_dto import UserDetailsDto
from news_recommender_backend.domain.global_state.global_state import GlobalState, get_global_state

def create_endpoint_auth():
    router_auth = APIRouter(
        prefix="/auth",
        tags=["auth"],
        responses={404: {"description": "Not found"}},
    )

    @router_auth.post("/fetchUserDetails")
    async def handle_fetch_user_details(request: Request) -> UserDetailsDto:
        logging.info("start fetch_user_details")
        firebase_user = auth_service.authenticate_user(request)
        global_state = get_global_state()

        userDetailsDto = await actions_auth.do_fetch_user_details(global_state, firebase_user)
        logging.info("end fetch_user_details")
        return userDetailsDto

    return router_auth