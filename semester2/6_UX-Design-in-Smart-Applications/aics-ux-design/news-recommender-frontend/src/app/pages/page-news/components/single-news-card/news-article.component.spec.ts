import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NewsArticleComponent } from './news-article.component';

describe('SingleNewsCardComponent', () => {
    let component: NewsArticleComponent;
    let fixture: ComponentFixture<NewsArticleComponent>;

    beforeEach(() => {
        TestBed.configureTestingModule({
            declarations: [NewsArticleComponent]
        });
        fixture = TestBed.createComponent(NewsArticleComponent);
        component = fixture.componentInstance;
        fixture.detectChanges();
    });

    it('should create', () => {
        expect(component).toBeTruthy();
    });
});
