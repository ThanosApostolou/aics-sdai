import { CategoriesEnum } from 'src/app/modules/core/enums/categories-enum';

export interface UserQuestionnaireDto {
    favorite_category: CategoriesEnum;
    is_extrovert: boolean;
    age: number;
    educational_level: number;
    sports_interest: number;
    arts_interest: number;
}

export interface AccountDetailsDto {
    name: string
    email: string
    user_questionnaire: UserQuestionnaireDto | null
}