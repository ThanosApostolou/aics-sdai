import { CategoriesEnum } from '../core/enums/categories-enum';

export interface UserDetailsDto {
    user_id: string;
    firebase_user_id: string;
    email: string;
    name: string;
    favorite_category: CategoriesEnum | null;
}