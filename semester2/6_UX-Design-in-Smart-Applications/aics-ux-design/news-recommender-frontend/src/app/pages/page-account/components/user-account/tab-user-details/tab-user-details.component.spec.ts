import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TabUserDetailsComponent } from './tab-user-details.component';

describe('TabUserDetailsComponent', () => {
  let component: TabUserDetailsComponent;
  let fixture: ComponentFixture<TabUserDetailsComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [TabUserDetailsComponent]
    });
    fixture = TestBed.createComponent(TabUserDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
