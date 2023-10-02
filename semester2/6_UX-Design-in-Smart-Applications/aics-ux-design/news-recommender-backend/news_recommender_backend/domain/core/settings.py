from functools import lru_cache
from dotenv import dotenv_values

class Settings:
    def __init__(self, front_url: str, db_host: str, db_schema: str, db_user: str, db_password: str, newsApiKey: str):
        self.front_url: str = front_url
        self.db_host: str = db_host
        self.db_schema: str = db_schema
        self.db_user: str = db_user
        self.db_password: str = db_password
        self.newsApiKey: str = newsApiKey

    def __repr__(self):
        return str(self.__dict__)



def create_settings() -> Settings:
    values = dotenv_values(".env")
    FRONT_URL = values["FRONT_URL"]
    DB_HOST = values["DB_HOST"]
    DB_SCHEMA = values["DB_SCHEMA"]
    DB_USER = values["DB_USER"]
    DB_PASSWORD = values["DB_PASSWORD"]
    NEWS_API_KEY = values["NEWS_API_KEY"]
    assert FRONT_URL
    assert DB_HOST
    assert DB_SCHEMA
    assert DB_USER
    assert DB_PASSWORD
    assert NEWS_API_KEY
    settings = Settings(FRONT_URL, DB_HOST, DB_SCHEMA, DB_USER, DB_PASSWORD, NEWS_API_KEY)
    return settings
