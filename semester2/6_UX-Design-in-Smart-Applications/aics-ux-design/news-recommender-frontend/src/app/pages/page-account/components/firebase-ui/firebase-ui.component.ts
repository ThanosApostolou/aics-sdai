import { AfterViewInit, Component } from '@angular/core';
import { Auth } from '@angular/fire/auth';
import { EmailAuthProvider } from 'firebase/auth';
import * as firebaseui from 'firebaseui';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'app-firebase-ui',
  templateUrl: './firebase-ui.component.html',
  styleUrls: ['./firebase-ui.component.scss'],
})
export class FirebaseUiComponent implements AfterViewInit {
  constructor(private auth: Auth) {}

  ngAfterViewInit(): void {
    var ui = new firebaseui.auth.AuthUI(this.auth);

    ui.start('#firebaseui-auth-container', {
      signInOptions: [EmailAuthProvider.PROVIDER_ID],
      signInSuccessUrl: environment.signInSuccessUrl,
      callbacks: {
        signInSuccessWithAuthResult: (
          authResult: any,
          redirectUrl?: string | undefined
        ) => {
          return this.signInSuccessWithAuthResult(authResult, redirectUrl);
        },
      },
    });
  }

  private signInSuccessWithAuthResult(
    authResult: any,
    redirectUrl?: string | undefined
  ): boolean {
    console.log('authResult', authResult);
    console.log('redirectUrl', redirectUrl);
    console.log('authResult.user', authResult.user);
    return true;
  }
}
