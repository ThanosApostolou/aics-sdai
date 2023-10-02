import { Component, Input, OnInit } from '@angular/core';
import { ArticleDto } from '../../dtos/fetch-news-dto';

@Component({
    selector: 'app-news-article',
    templateUrl: './news-article.component.html',
    styleUrls: ['./news-article.component.scss']
})
export class NewsArticleComponent implements OnInit {
    @Input()
    articleDto!: ArticleDto;

    constructor() { }

    ngOnInit(): void {

    }

}
