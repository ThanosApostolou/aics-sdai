from pathlib import Path
from fastapi import FastAPI
from functools import lru_cache
from sqlalchemy import Engine
import firebase_admin

from news_recommender_backend.domain.core.settings import Settings


class GlobalState:
    def __init__(self, api: FastAPI | None, settings: Settings, db_engine: Engine, firebaseApp: firebase_admin.App):
        self.api = api
        self.settings = settings
        self.db_engine = db_engine
        self.firebaseApp = firebaseApp
        self.rootPath = Path().absolute()



__global_state: GlobalState | None = None


def initialize_global_state(api: FastAPI | None, settings: Settings, db_engine: Engine, firebaseApp: firebase_admin.App):
    global __global_state
    __global_state = GlobalState(api, settings, db_engine, firebaseApp)


@lru_cache()
def get_global_state() -> GlobalState:
    assert __global_state is not None
    return __global_state