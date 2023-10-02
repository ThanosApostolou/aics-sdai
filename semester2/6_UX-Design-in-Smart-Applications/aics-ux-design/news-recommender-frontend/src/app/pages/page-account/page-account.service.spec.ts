import { TestBed } from '@angular/core/testing';

import { PageAccountService } from './page-account.service';

describe('PageAccountService', () => {
  let service: PageAccountService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PageAccountService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
