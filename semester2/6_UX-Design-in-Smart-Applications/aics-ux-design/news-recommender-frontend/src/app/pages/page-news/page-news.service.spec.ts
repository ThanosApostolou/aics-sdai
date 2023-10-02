import { TestBed } from '@angular/core/testing';

import { PageNewsService } from './page-news.service';

describe('PageNewsService', () => {
  let service: PageNewsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PageNewsService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
