import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { UserDetailsDto } from 'src/app/modules/auth/user-details-dto';
import { environment } from 'src/environments/environment';
import { AccountDetailsDto } from './dtos/account-details-dto';

@Injectable({
    providedIn: 'root'
})
export class PageAccountService {
    constructor(private httpClient: HttpClient) { }

    fetchAccountDetails(): Observable<AccountDetailsDto> {
        const fetchAccountDetailsUrl = environment.BACKEND_URL + '/account/fetchAccountDetails';
        return this.httpClient.get<AccountDetailsDto>(fetchAccountDetailsUrl, {});
    }

    saveAccountDetails(accountDetailsDto: AccountDetailsDto): Observable<string[]> {
        const saveAccountDetailsUrl = environment.BACKEND_URL + '/account/saveAccountDetails';
        return this.httpClient.post<string[]>(saveAccountDetailsUrl, accountDetailsDto);

    }

}
