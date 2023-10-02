import { TestBed } from '@angular/core/testing';

import { AppStarterService } from './app-starter.service';

describe('AppStarterService', () => {
  let service: AppStarterService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AppStarterService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
