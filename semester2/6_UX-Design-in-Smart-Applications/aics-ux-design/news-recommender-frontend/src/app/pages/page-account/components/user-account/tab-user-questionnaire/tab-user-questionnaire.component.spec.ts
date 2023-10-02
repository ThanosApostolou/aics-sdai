import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TabUserQuestionnaireComponent } from './tab-user-questionnaire.component';

describe('TabUserQuestionnaireComponent', () => {
  let component: TabUserQuestionnaireComponent;
  let fixture: ComponentFixture<TabUserQuestionnaireComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [TabUserQuestionnaireComponent]
    });
    fixture = TestBed.createComponent(TabUserQuestionnaireComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
