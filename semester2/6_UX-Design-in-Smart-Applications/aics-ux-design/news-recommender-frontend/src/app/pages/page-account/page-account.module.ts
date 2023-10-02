import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { PageAccountRoutingModule } from './page-account-routing.module';
import { PageAccountComponent } from './page-account.component';
import { CoreModule } from 'src/app/modules/core/core.module';
import { FirebaseUiComponent } from './components/firebase-ui/firebase-ui.component';
import { UserAccountComponent } from './components/user-account/user-account.component';
import { TabUserDetailsComponent } from './components/user-account/tab-user-details/tab-user-details.component';
import { TabUserQuestionnaireComponent } from './components/user-account/tab-user-questionnaire/tab-user-questionnaire.component';

@NgModule({
  declarations: [
    PageAccountComponent,
    FirebaseUiComponent,
    UserAccountComponent,
    TabUserDetailsComponent,
    TabUserQuestionnaireComponent,
  ],
  imports: [CommonModule, PageAccountRoutingModule, CoreModule],
})
export class PageAccountModule {}
