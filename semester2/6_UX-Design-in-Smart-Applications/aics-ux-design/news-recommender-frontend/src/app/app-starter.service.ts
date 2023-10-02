import { Injectable } from '@angular/core';
import { Auth } from '@angular/fire/auth';
import { GlobalStateStoreService } from './modules/core/global-state-store.service';
import { AuthService } from './modules/auth/auth.service';

@Injectable({
    providedIn: 'root'
})
export class AppStarterService {

    constructor(private auth: Auth,
        private globalStateStoreService: GlobalStateStoreService,
        private authService: AuthService
    ) { }

    init(): Promise<void> {
        const promise = new Promise<void>((resolve, reject) => {
            console.log('app started')
            this.auth.onAuthStateChanged(firebaseUser => {
                console.log('init user', firebaseUser)
                this.globalStateStoreService.firebaseUser.set(firebaseUser)
                if (firebaseUser) {
                    this.authService.fetchUserDetails().subscribe({
                        next: (userDetailsDto) => {
                            console.log('starter userDetailsDto', userDetailsDto)
                            this.globalStateStoreService.userDetails.set(userDetailsDto)
                            resolve();
                        },
                        error: (e) => {
                            reject(e);
                        }
                    })
                } else {
                    resolve();
                }
            }, error => {
                console.log('init error', error)
                this.globalStateStoreService.firebaseUser.set(null)
                reject();
            })

        });
        return promise;
    }
}
