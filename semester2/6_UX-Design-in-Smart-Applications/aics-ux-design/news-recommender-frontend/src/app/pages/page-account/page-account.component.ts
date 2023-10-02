import { Component, OnInit, AfterViewInit } from '@angular/core';
import { PagesEnum } from 'src/app/modules/core/enums/pages-enum';
import { GlobalStateStoreService } from 'src/app/modules/core/global-state-store.service';
import { Auth, EmailAuthProvider } from '@angular/fire/auth';
import * as firebaseui from 'firebaseui';
import { environment } from 'src/environments/environment';
import { PageAccountService } from './page-account.service';

@Component({
    selector: 'app-page-account',
    templateUrl: './page-account.component.html',
    styleUrls: ['./page-account.component.scss']
})
export class PageAccountComponent implements OnInit {
    public firebaseUser = this.globalStateStoreService.firebaseUser;

    constructor(private globalStateStoreService: GlobalStateStoreService,
        private auth: Auth,
        private pageAccountService: PageAccountService) {
    }

    ngOnInit() {
        setTimeout(() => {
            this.globalStateStoreService.activePage.set(PagesEnum.ACCOUNT)
        });
    }
}
