import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { CategoriesEnum } from 'src/app/modules/core/enums/categories-enum';
import { environment } from 'src/environments/environment';
import { FetchNewsRequestDto, FetchNewsResponseDto } from './dtos/fetch-news-dto';

@Injectable({
    providedIn: 'root'
})
export class PageNewsService {

    constructor(private httpClient: HttpClient) { }


    fetchNews(fetchNewsRequestDto: FetchNewsRequestDto): Observable<FetchNewsResponseDto> {
        console.log('fetchNewsRequestDto', fetchNewsRequestDto)
        const fetchNewsUrl = environment.BACKEND_URL + '/news/fetchNews';
        let queryParams = new HttpParams();
        if (fetchNewsRequestDto.category) {
            queryParams = queryParams.append("category", fetchNewsRequestDto.category)
        }
        if (fetchNewsRequestDto.country) {
            queryParams = queryParams.append("country", fetchNewsRequestDto.country)
        }
        if (fetchNewsRequestDto.query) {
            queryParams = queryParams.append("query", fetchNewsRequestDto.query)
        }
        console.log('queryParams', queryParams)

        return this.httpClient.get<FetchNewsResponseDto>(fetchNewsUrl, {
            params: queryParams
        });
    }

    recommendCategory(): Observable<CategoriesEnum> {
        const recommendCategoryUrl = environment.BACKEND_URL + '/news/recommendCategory';
        return this.httpClient.get<CategoriesEnum>(recommendCategoryUrl);
    }
}
