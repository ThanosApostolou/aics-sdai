from pydantic import BaseModel


class UserQuestionnaireDto(BaseModel):
    favorite_category: str
    is_extrovert:bool
    age: int
    educational_level: int
    sports_interest: int
    arts_interest: int


class AccountDetailsDto(BaseModel):
    name: str = ''
    email: str = ''
    user_questionnaire: UserQuestionnaireDto | None