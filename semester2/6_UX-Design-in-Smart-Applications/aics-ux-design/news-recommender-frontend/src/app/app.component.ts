import { Component, WritableSignal } from '@angular/core';
import { SidenavLink } from './modules/core/models/sidenav-link';
import { PagesEnum } from './modules/core/enums/pages-enum';
import { GlobalStateStoreService } from './modules/core/global-state-store.service';
import { AuthService } from './modules/auth/auth.service';
import { Router } from '@angular/router';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss']
})
export class AppComponent {
    private title = 'News Recommender';
    readonly PagesEnum = PagesEnum;

    public opened = false;
    public sidenavLinks: SidenavLink[] = [
        {
            name: PagesEnum.HOME,
            label: 'Home',
            path: '/home',
        },
        {
            name: PagesEnum.NEWS,
            label: 'News',
            path: '/news',
        }
    ];

    public activePage: WritableSignal<PagesEnum> = this.globalStateStoreService.activePage;
    public userDetails = this.globalStateStoreService.userDetails;

    constructor(private globalStateStoreService: GlobalStateStoreService,
        private authService: AuthService,
        private router: Router) { }

    public sidebarButtonClicked() {
        this.opened = !this.opened;
    }

    public async signOutClicked() {
        await this.authService.signOut();
        window.location.reload();
    }
}
