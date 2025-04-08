import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NewDoctorFormComponent } from './new-doctor-form.component';

describe('NewDoctorFormComponent', () => {
  let component: NewDoctorFormComponent;
  let fixture: ComponentFixture<NewDoctorFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ NewDoctorFormComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(NewDoctorFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
