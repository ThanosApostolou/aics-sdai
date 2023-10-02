import { TestBed } from '@angular/core/testing';

import { GlobalStateStoreService } from './global-state-store.service';

describe('GlobalStateStoreService', () => {
  let service: GlobalStateStoreService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(GlobalStateStoreService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
