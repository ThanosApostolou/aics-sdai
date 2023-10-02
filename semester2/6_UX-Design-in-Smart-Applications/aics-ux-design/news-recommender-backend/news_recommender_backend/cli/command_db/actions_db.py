from news_recommender_backend.domain.entities.entities import Base
from news_recommender_backend.domain.global_state.global_state import get_global_state


def do_init_db():
    golbal_state = get_global_state()
    Base.metadata.create_all(golbal_state.db_engine)