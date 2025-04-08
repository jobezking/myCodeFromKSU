import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NewConditionFormComponent } from './new-condition-form.component';

describe('NewConditionFormComponent', () => {
  let component: NewConditionFormComponent;
  let fixture: ComponentFixture<NewConditionFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ NewConditionFormComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(NewConditionFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
