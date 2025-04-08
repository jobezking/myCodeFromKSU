import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ListConditionsComponent } from './list-conditions.component';

describe('ListConditionsComponent', () => {
  let component: ListConditionsComponent;
  let fixture: ComponentFixture<ListConditionsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ListConditionsComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ListConditionsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
