
from fastapi import FastAPI
from fastapi import APIRouter
from news_recommender_backend.api.endpoint_account.endpoint_account import create_endpoint_account
from news_recommender_backend.api.endpoint_auth.endpoint_auth import create_endpoint_auth
from news_recommender_backend.api.endpoint_news.endpoint_news import create_endpoint_news
import logging
from fastapi.middleware.cors import CORSMiddleware

from news_recommender_backend.domain.global_state.global_state import get_global_state


def create_api():
    api: FastAPI = FastAPI()
    # cors
    origins = [
        "http://localhost:4200",
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1:4200",
    ]
    api.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
    )
    # routers
    api.include_router(create_endpoint_auth())
    api.include_router(create_endpoint_news())
    api.include_router(create_endpoint_account())


    @api.get("/")
    async def root():
        logging.info("settings: " + str(get_global_state().settings))
        return {"message": "hello"}

    return api