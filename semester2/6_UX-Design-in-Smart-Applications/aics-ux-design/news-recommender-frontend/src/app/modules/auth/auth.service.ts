import { Injectable } from '@angular/core';
import { FirebaseOptions, initializeApp } from 'firebase/app';
import { getAnalytics } from 'firebase/analytics';
import { getAuth } from "firebase/auth";
import { environment } from 'src/environments/environment';
import { Auth } from '@angular/fire/auth';
import { GlobalStateStoreService } from '../core/global-state-store.service';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { UserDetailsDto } from './user-details-dto';

@Injectable({
    providedIn: 'root',
})
export class AuthService {
    constructor(private auth: Auth,
        private globalStateStoreService: GlobalStateStoreService,
        private httpClient: HttpClient) { }

    fetchUserDetails(): Observable<UserDetailsDto> {
        const fetchUserDetailsUrl = environment.BACKEND_URL + '/auth/fetchUserDetails';
        const headers = this.globalStateStoreService.firebaseUser().accessToken
            ? {
                'Authorization': this.globalStateStoreService.firebaseUser().accessToken
            }
            : undefined
        return this.httpClient.post<UserDetailsDto>(fetchUserDetailsUrl, {}, {
            headers
        });

    }

    async signOut(): Promise<void> {
        await this.auth.signOut();
        this.globalStateStoreService.firebaseUser.set(null);
        this.globalStateStoreService.userDetails.set(null);
    }
}
