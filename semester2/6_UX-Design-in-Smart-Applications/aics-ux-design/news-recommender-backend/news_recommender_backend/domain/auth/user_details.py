from news_recommender_backend.domain.news.news_constants import NewsCategoriesEnum


class UserDetails:
    def __init__(self, user_id: int, firebase_user_id: str, email: str, name: str, favorite_category: NewsCategoriesEnum | None):
        self.user_id = user_id
        self.firebase_user_id = firebase_user_id
        self.email = email
        self.name = name
        self.favorite_category = favorite_category


    def __repr__(self):
        return str(self.__dict__)