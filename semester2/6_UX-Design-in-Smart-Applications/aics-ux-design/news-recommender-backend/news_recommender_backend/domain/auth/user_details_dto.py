from pydantic import BaseModel

from news_recommender_backend.domain.auth.user_details import UserDetails
from news_recommender_backend.domain.news.news_constants import NewsCategoriesEnum

class UserDetailsDto(BaseModel):
    user_id: int = 0
    firebase_user_id: str = ''
    email: str = ''
    name: str = ''
    favorite_category: NewsCategoriesEnum | None

    @staticmethod
    def fromUserDetails(user_details: UserDetails) -> 'UserDetailsDto':
        return UserDetailsDto(user_id=user_details.user_id, firebase_user_id=user_details.firebase_user_id, email=user_details.email, name=user_details.name, favorite_category=user_details.favorite_category)