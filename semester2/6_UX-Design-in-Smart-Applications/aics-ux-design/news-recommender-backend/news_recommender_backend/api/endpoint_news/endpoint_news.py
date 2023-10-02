
import logging
import traceback
from fastapi import APIRouter, HTTPException, Request
from news_recommender_backend.api.endpoint_news import actions_news
import news_recommender_backend.domain.auth.auth_service as auth_service
import requests

from news_recommender_backend.domain.auth.user_details_dto import UserDetailsDto
from news_recommender_backend.domain.global_state.global_state import get_global_state
from news_recommender_backend.domain.news.news_constants import NewsCategoriesEnum

def create_endpoint_news():
    endpoint_news = APIRouter(
        prefix="/news",
        tags=["news"],
        responses={404: {"description": "Not found"}},
    )


    @endpoint_news.get("/fetchNews")
    async def handle_fetch_news(request: Request, category: NewsCategoriesEnum | None = None, country: str | None = None, query: str="") -> dict:
        logging.info(f"category, country, query = {category, country, query}")
        logging.info('start handle_fetch_news')
        global_state = get_global_state()
        newsapi_key = global_state.settings.newsApiKey
        url = f"https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": newsapi_key
        }
        if query is not None:
            params["q"] = query
        if category is not None:
            params["category"] = category
        if country is not None:
            params["country"] = country

        r = requests.get(url=url, params=params)
        # extracting data in json format
        data = r.json()
        logging.info(f'data {data}')
        logging.info('end handle_fetch_news')
        return data


    @endpoint_news.get("/fetchNewsDetails")
    async def handle_fetch_news_details(request: Request) -> dict:
        # TODO see if this is valid
        global_state = get_global_state()
        new_api_key = global_state.settings.newsApiKey
        print('new_api_key', new_api_key)
        return {"fetchNewsDetails": "fetchNewsDetails"}



    @endpoint_news.get("/recommendCategory")
    async def handle_recommend_category(request: Request) -> NewsCategoriesEnum:
        try:
            globalState = get_global_state()
            user_details = auth_service.authorize_user(request)
            recommended_category: NewsCategoriesEnum = actions_news.do_recommend_category(globalState, user_details)
            return recommended_category
        except Exception as error:
            logging.error(error)
            traceback.print_exc()
            raise HTTPException(status_code=500)

    return endpoint_news