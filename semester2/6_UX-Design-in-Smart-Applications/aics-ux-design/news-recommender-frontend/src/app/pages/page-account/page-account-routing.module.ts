import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PageAccountComponent } from './page-account.component';

const routes: Routes = [
  {
    path: '',
    component: PageAccountComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class PageAccountRoutingModule { }
