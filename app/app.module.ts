import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HttpClientModule } from '@angular/common/http';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input'
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatMenuModule } from '@angular/material/menu';
import { MatIconModule } from '@angular/material/icon';
import { Routes, RouterModule } from '@angular/router';

import { ConditionService } from './condition.service';
import { PatientService } from './patient.service';
import { StudentService } from './student.service';
import { DoctorService } from './doctor.service';

import { NewConditionFormComponent } from './new-condition-form/new-condition-form.component';
import { NavigationMenuComponent } from './navigation-menu/navigation-menu.component';
import { NotFoundComponent } from './not-found/not-found.component';
import { ListConditionsComponent } from './list-conditions/list-conditions.component';
import { ListPatientsComponent } from './list-patients/list-patients.component';
import { NewAppointmentFormComponent } from './new-appointment-form/new-appointment-form.component';
import { ListStudentsComponent } from './list-students/list-students.component';
import { NewStudentFormComponent } from './new-student-form/new-student-form.component';
import { ListDoctorsComponent } from './list-doctors/list-doctors.component';
import { NewDoctorFormComponent } from './new-doctor-form/new-doctor-form.component';
import { LoginComponent } from './login/login.component';

import { ReactiveFormsModule } from '@angular/forms';
import { SocialLoginModule, SocialAuthServiceConfig } from 'angularx-social-login';
import { GoogleLoginProvider } from 'angularx-social-login';
import { MainpageComponent } from './mainpage/mainpage.component';


const appRoutes: Routes = [ 
  { path: '',  component: LoginComponent }, //default component to display
  { path: 'addCondition',   component: NewConditionFormComponent }, //when Condition added
  { path: 'editCondition/:_id', component: NewConditionFormComponent }, //when Condition edited 
  { path: 'listConditions',  component: ListConditionsComponent }, //when Conditions listed
  { path: 'addPatient',   component: NewAppointmentFormComponent }, //when Patient added
  { path: 'editPatient/:_id', component: NewAppointmentFormComponent }, //when Patient edited
  { path: 'listPatients',  component: ListPatientsComponent }, //when Conditions listed
  { path: 'addStudent',   component: NewStudentFormComponent }, //when Patient added
  { path: 'editStudent/:_id', component: NewStudentFormComponent }, //when Patient edited
  { path: 'listStudents',  component: ListStudentsComponent }, //when Conditions listed
  { path: 'addDoctor',   component: NewDoctorFormComponent }, //when Patient added
  { path: 'editDoctor/:_id', component: NewDoctorFormComponent }, //when Patient edited
  { path: 'listDoctors',  component: ListDoctorsComponent }, //when Conditions listed
  { path: 'mainpage',  component: MainpageComponent }, //app main page
  { path: '**',   component: NotFoundComponent } //when path cannot be found
];

@NgModule({
  declarations: [
    AppComponent,
    NewConditionFormComponent,
    NavigationMenuComponent,
    NotFoundComponent,
    ListConditionsComponent,
    ListPatientsComponent,
    NewAppointmentFormComponent,
    ListStudentsComponent,
    NewStudentFormComponent,
    ListDoctorsComponent,
    NewDoctorFormComponent,
    LoginComponent,
    MainpageComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    MatFormFieldModule,
    MatInputModule,
    BrowserAnimationsModule,
    FormsModule,
    MatButtonModule,
    MatMenuModule,
    MatIconModule,
    ReactiveFormsModule,
    SocialLoginModule,
    RouterModule.forRoot(appRoutes)
  ],
  providers: [ConditionService, PatientService, StudentService, DoctorService,
    {
      provide: 'SocialAuthServiceConfig',
      useValue: {
        autoLogin: false,
        providers: [
          {
            id: GoogleLoginProvider.PROVIDER_ID,
            provider: new GoogleLoginProvider(
              '263975214940-9etiqajk1ulfh2ol6giqif8ckrapldd3.apps.googleusercontent.com'
            )
          }
        ]
      } as SocialAuthServiceConfig,
    }    
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
