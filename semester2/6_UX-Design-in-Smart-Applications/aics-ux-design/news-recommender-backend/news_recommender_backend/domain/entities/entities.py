from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import ForeignKey, String, Integer, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_recommender_backend.domain.news.news_constants import NewsCategoriesEnum


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(primary_key=True)
    firebase_user_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=False)
    questionnaire: Mapped["UserQuestionnaire"] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="joined"
    )

    def __repr__(self) -> str:
        return f"User(id={self.user_id!r}, firebase_user_id={self.firebase_user_id!r}, email={self.email!r}, name={self.name!r})"

class UserQuestionnaire(Base):
    __tablename__ = "user_questionnaires"
    user_questionnaire_id: Mapped[int] = mapped_column(primary_key=True)
    favorite_category: Mapped[NewsCategoriesEnum] = mapped_column(String(255), nullable=False)
    is_extrovert: Mapped[bool] = mapped_column(Boolean(), nullable=False)
    age: Mapped[int] = mapped_column(Integer(), nullable=False)
    educational_level: Mapped[int] = mapped_column(Integer(), nullable=False)
    sports_interest: Mapped[int] = mapped_column(Integer(), nullable=False)
    arts_interest: Mapped[int] = mapped_column(Integer(), nullable=False)

    user_id_fk: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False, unique=True)
    user: Mapped["User"] = relationship(back_populates="questionnaire")

    def __repr__(self) -> str:
        return f"UserQuestionnaire(user_questionnaire_id={self.user_questionnaire_id!r}, is_extrovert={self.is_extrovert!r}, age={self.age!r}, educational_level={self.educational_level!r}, sports_interest={self.sports_interest!r}, user_id_fk={self.user_id_fk!r})"


class NnModel(Base):
    __tablename__ = "nn_models"
    nn_model_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    path: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    is_trained: Mapped[bool] = mapped_column(Boolean(), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean(), nullable=False)

    def __repr__(self) -> str:
        return f"NnModel(nn_model_id={self.nn_model_id!r}, name={self.name!r}, path={self.path!r}, is_trained={self.is_trained!r}, is_active={self.is_active!r})"