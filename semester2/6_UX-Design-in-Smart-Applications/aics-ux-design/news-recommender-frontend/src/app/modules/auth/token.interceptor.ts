import { Injectable } from '@angular/core';
import {
    HttpRequest,
    HttpHandler,
    HttpEvent,
    HttpInterceptor
} from '@angular/common/http';
import { Observable } from 'rxjs';
import { GlobalStateStoreService } from '../core/global-state-store.service';

@Injectable()
export class TokenInterceptor implements HttpInterceptor {

    constructor(private globalStateStoreService: GlobalStateStoreService) { }

    intercept(request: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
        const newRequest = this.globalStateStoreService.firebaseUser().accessToken
            ? request.clone({
                setHeaders: {
                    'Authorization': this.globalStateStoreService.firebaseUser().accessToken
                }
            }) : request.clone()
        return next.handle(newRequest);
    }
}
