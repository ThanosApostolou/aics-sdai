import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PageNewsComponent } from './page-news.component';

const routes: Routes = [
  {
    path: '',
    component: PageNewsComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class PageNewsRoutingModule { }
