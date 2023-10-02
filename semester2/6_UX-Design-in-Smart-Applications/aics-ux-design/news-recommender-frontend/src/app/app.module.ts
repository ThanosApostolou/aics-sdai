import { APP_INITIALIZER, NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { CoreModule } from './modules/core/core.module';
import { provideFirebaseApp, initializeApp } from '@angular/fire/app';
import { Auth, getAuth, provideAuth } from '@angular/fire/auth';
import { environment } from 'src/environments/environment';
import { AppStarterService } from './app-starter.service';
import { HTTP_INTERCEPTORS, HttpClientModule } from '@angular/common/http';
import { GlobalStateStoreService } from './modules/core/global-state-store.service';
import { AuthService } from './modules/auth/auth.service';
import { TokenInterceptor } from './modules/auth/token.interceptor';

@NgModule({
    declarations: [
        AppComponent
    ],
    imports: [
        BrowserModule,
        AppRoutingModule,
        BrowserAnimationsModule,
        CoreModule,
        HttpClientModule,
        provideFirebaseApp(() => initializeApp(environment.firebaseOptions)),
        provideAuth(() => getAuth()),
    ],
    providers: [
        {
            provide: HTTP_INTERCEPTORS,
            useClass: TokenInterceptor,
            multi: true
        },
        {
            provide: APP_INITIALIZER,
            useFactory: (appStarterService: AppStarterService) => (() => appStarterService.init()),
            deps: [AppStarterService, Auth, GlobalStateStoreService, AuthService],
            multi: true
        }
    ],
    bootstrap: [AppComponent]
})
export class AppModule { }
