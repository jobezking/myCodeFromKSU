import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ReactiveFormsModule } from '@angular/forms';

import { HttpClientModule } from '@angular/common/http';
import { CapstoneService } from './capstone.service';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatButtonModule } from '@angular/material/button';
import { FormsModule } from '@angular/forms';
import { MatMenuModule } from '@angular/material/menu';
import { MatIconModule } from '@angular/material/icon';
import { Routes, RouterModule } from '@angular/router';

import { ProposalComponent } from './proposal/proposal.component';
import { NewProjectFormComponent } from './new-project-form/new-project-form.component';
import { NavigationMenuComponent } from './navigation-menu/navigation-menu.component';
import { NotFoundComponent } from './not-found/not-found.component';
import { ListProjectsComponent } from './list-projects/list-projects.component';

const appRoutes: Routes = [ {
  path: '',  //default component to display
  component: ListProjectsComponent
}, {
  path: 'addProject',  //when project added 
  component: NewProjectFormComponent
}, {
  path: 'editProject/:_id', //when students edited 
  component: NewProjectFormComponent
},{
  path: 'listProjects',  //when projects listed
  component: ListProjectsComponent
}, {
  path: 'showProposal',  //display proposal form 
  component: ProposalComponent
},{
  path: '**',  //when path cannot be found
  component: NotFoundComponent
}
];

@NgModule({
  declarations: [
    AppComponent,
    ProposalComponent,
    NewProjectFormComponent,
    NavigationMenuComponent,
    NotFoundComponent,
    ListProjectsComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    ReactiveFormsModule,
    HttpClientModule,
    MatFormFieldModule,
    MatInputModule,
    BrowserAnimationsModule,
    MatButtonModule,
    FormsModule,
    MatMenuModule,
    MatIconModule,
    RouterModule.forRoot(appRoutes)
  ],
  providers: [CapstoneService],
  bootstrap: [AppComponent]
})
export class AppModule { }

