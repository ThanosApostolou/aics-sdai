import { Component, OnInit, WritableSignal } from '@angular/core';
import { PagesEnum } from 'src/app/modules/core/enums/pages-enum';
import { GlobalStateStoreService } from 'src/app/modules/core/global-state-store.service';
import { PageNewsService } from './page-news.service';
import { CategoriesEnum } from 'src/app/modules/core/enums/categories-enum';
import { FetchNewsResponseDto } from './dtos/fetch-news-dto';
import { FormGroup, FormControl } from '@angular/forms';
import { CountriesEnum } from 'src/app/modules/core/enums/countries-enum copy';
import { UserDetailsDto } from 'src/app/modules/auth/user-details-dto';

@Component({
    selector: 'app-page-news',
    templateUrl: './page-news.component.html',
    styleUrls: ['./page-news.component.scss']
})
export class PageNewsComponent implements OnInit {
    fetchNewsResponse: FetchNewsResponseDto | null = null;

    CategoriesEnum = CategoriesEnum;
    allCategories = Object.values(CategoriesEnum);

    CountriesEnum = CountriesEnum;
    allCountries = Object.values(CountriesEnum);

    searchFormGroup: FormGroup = new FormGroup({});
    hasError: boolean = false;
    isWaitingAsyncResult: boolean = false;
    userDetails: WritableSignal<UserDetailsDto | null> = this.globalStateStoreService.userDetails;

    constructor(private globalStateStoreService: GlobalStateStoreService,
        private pageNewsService: PageNewsService
    ) {
    }

    ngOnInit() {
        setTimeout(() => {
            this.globalStateStoreService.activePage.set(PagesEnum.NEWS)
        })

        const disabled = false;
        this.searchFormGroup = new FormGroup({
            category: new FormControl<CategoriesEnum | null>({ value: CategoriesEnum.general, disabled }),
            country: new FormControl<CountriesEnum | null>({ value: CountriesEnum.us, disabled }),
            query: new FormControl<string>({ value: "", disabled }),
        });
        this.userDetails = this.globalStateStoreService.userDetails;
    }

    loadData() {
        const category = this.searchFormGroup?.get('category')?.value;
        const country = this.searchFormGroup?.get('country')?.value;
        const query = this.searchFormGroup?.get('query')?.value;

        this.isWaitingAsyncResult = true;
        this.pageNewsService.fetchNews({
            category,
            country,
            query
        }).subscribe({
            next: (result) => {
                console.log('result', result)
                if (result.status === 'ok') {
                    this.hasError = false;
                    this.fetchNewsResponse = result;
                } else {
                    this.handleError();
                }
                this.isWaitingAsyncResult = false;
            },
            error: (error) => {
                console.error('error', error)
                this.handleError();
                this.isWaitingAsyncResult = false;
            }
        });
    }

    private handleError() {
        this.fetchNewsResponse = null;
        this.hasError = true;
    }

    recommendCategory() {
        this.isWaitingAsyncResult = true;
        this.pageNewsService.recommendCategory().subscribe({
            next: (result) => {
                this.searchFormGroup?.get('category')?.setValue(result)
                this.loadData()
            },
            error: (error) => {
                this.isWaitingAsyncResult = false;
                console.error('error', error)
            }
        })
    }
}
