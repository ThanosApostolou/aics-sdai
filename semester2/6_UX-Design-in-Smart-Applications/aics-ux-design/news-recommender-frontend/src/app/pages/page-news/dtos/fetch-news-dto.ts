import { CategoriesEnum } from 'src/app/modules/core/enums/categories-enum';
import { CountriesEnum } from 'src/app/modules/core/enums/countries-enum copy';

export interface FetchNewsRequestDto {
    category: CategoriesEnum;
    country: CountriesEnum;
    query: string;
}

export interface FetchNewsResponseDto {
    status: string;
    totalResults: number;
    articles: ArticleDto[];
}

export interface ArticleDto {
    source: {
        id: string;
        name: string;
    };
    author: string;
    title: string;
    description: string;
    url: string;
    urlToImage: string;
    publishedAt: string;
    content: string

}