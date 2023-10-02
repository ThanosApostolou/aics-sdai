import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { PageNewsRoutingModule } from './page-news-routing.module';
import { PageNewsComponent } from './page-news.component';
import { CoreModule } from 'src/app/modules/core/core.module';
import { NewsArticleComponent } from './components/single-news-card/news-article.component';


@NgModule({
    declarations: [
        PageNewsComponent,
        NewsArticleComponent
    ],
    imports: [
        CommonModule,
        PageNewsRoutingModule,
        CoreModule
    ]
})
export class PageNewsModule { }
