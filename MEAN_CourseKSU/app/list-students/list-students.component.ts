import { Component, OnInit } from '@angular/core';
import { StudentService } from '../student.service';

@Component({
  selector: 'app-list-students',
  templateUrl: './list-students.component.html',
  styleUrls: ['./list-students.component.css']
})
export class ListStudentsComponent implements OnInit {
  //declare variable to hold response and make it public to be accessible from components.html
  public students: any;
  //initialize the call using StudentService 
  constructor(private _myService: StudentService) { }
  ngOnInit() {
      this.getStudents();
  }
  //method called OnInit
  
  getStudents() {
      this._myService.getStudents().subscribe(
          //read data and assign to public variable students
          data => { this.students = data},
          err => console.error(err),
          () => console.log('finished loading')
      );
  
   
  }
  onDelete(studentId: string) {
    this._myService.deleteStudent(studentId);
    location.reload();
  }

}
