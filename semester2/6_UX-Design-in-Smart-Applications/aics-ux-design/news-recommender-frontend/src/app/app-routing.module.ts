import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { authorizationGuard } from './modules/auth/authorization.guard';

const routes: Routes = [
    { path: 'home', canActivate: [authorizationGuard], loadChildren: () => import('./pages/page-home/page-home.module').then(m => m.PageHomeModule) },
    { path: 'account', canActivate: [authorizationGuard], loadChildren: () => import('./pages/page-account/page-account.module').then(m => m.PageAccountModule) },
    { path: 'news', canActivate: [authorizationGuard], loadChildren: () => import('./pages/page-news/page-news.module').then(m => m.PageNewsModule) },
    { path: '', redirectTo: '/home', pathMatch: 'full' }, // redirect to home
    { path: '**', loadChildren: () => import('./pages/page-not-found/page-not-found.module').then(m => m.PageNotFoundModule) },  // Wildcard route for a 404 page
];

@NgModule({
    imports: [RouterModule.forRoot(routes)],
    exports: [RouterModule]
})
export class AppRoutingModule { }
