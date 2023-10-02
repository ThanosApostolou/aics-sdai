from sqlalchemy import Engine, create_engine

from news_recommender_backend.domain.core.settings import Settings

def create_db_engine(settings: Settings) -> Engine:
    db_user = settings.db_user
    db_password = settings.db_password
    db_host = settings.db_host
    db_schema = settings.db_schema
    connection_str = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_schema}?charset=utf8mb4"
    db_engine = create_engine(connection_str, echo=True)
    return db_engine