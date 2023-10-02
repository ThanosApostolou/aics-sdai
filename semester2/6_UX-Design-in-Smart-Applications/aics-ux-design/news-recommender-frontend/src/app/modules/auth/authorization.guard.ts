import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from './auth.service';
import { inject } from '@angular/core';
import { GlobalStateStoreService } from '../core/global-state-store.service';
import { PagesEnum } from '../core/enums/pages-enum';

export const authorizationGuard: CanActivateFn = async (route, state) => {
    const globalStateStoreService: GlobalStateStoreService = inject(GlobalStateStoreService);
    const router: Router = inject(Router);
    console.log('route.url', route.url)

    if (globalStateStoreService.userDetails() == null
        && route.url?.[0].path != 'account'
    ) {
        await router.navigate(['/account']);
        return false;
    }
    return true;
};
