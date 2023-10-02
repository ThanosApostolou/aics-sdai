import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { PageHomeRoutingModule } from './page-home-routing.module';
import { PageHomeComponent } from './page-home.component';
import { CoreModule } from 'src/app/modules/core/core.module';


@NgModule({
  declarations: [

    PageHomeComponent
  ],
  imports: [
    CommonModule,
    PageHomeRoutingModule,
    CoreModule
  ]
})
export class PageHomeModule { }
