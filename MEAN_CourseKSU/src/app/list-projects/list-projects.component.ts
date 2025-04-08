import { Component, OnInit } from '@angular/core';
import { CapstoneService } from '../capstone.service';

@Component({
  selector: 'app-list-projects',
  templateUrl: './list-projects.component.html',
  styleUrls: ['./list-projects.component.css']
})
export class ListProjectsComponent implements OnInit {
  //declare variable to hold response and make it public to be accessible from components.html
  public projects: any;
  //initialize the call using StudentService 
  constructor(private _myService: CapstoneService) { }
  ngOnInit() {
      this.getProjects();
  }
  //method called OnInit
  getProjects() {
      this._myService.getProjects().subscribe(
          //read data and assign to public variable students
          data => { this.projects = data},
          err => console.error(err),
          () => console.log('finished loading')
      );
  }
  onDelete(projectId: string) {
    this._myService.deleteProject(projectId);
  }
}
