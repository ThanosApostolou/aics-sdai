import { Injectable, WritableSignal, signal } from '@angular/core';
import { PagesEnum } from './enums/pages-enum';
import { UserDetailsDto } from '../auth/user-details-dto';

@Injectable({
    providedIn: 'root'
})
export class GlobalStateStoreService {
    public readonly activePage: WritableSignal<PagesEnum> = signal(PagesEnum.HOME);
    public readonly firebaseUser: WritableSignal<any> = signal(null);
    public readonly userDetails: WritableSignal<UserDetailsDto | null> = signal(null);

    constructor() { }

}
