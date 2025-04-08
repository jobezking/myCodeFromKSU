import { Component, OnInit, Input,  } from '@angular/core';
import { StudentService } from '../student.service';
import {  Router} from '@angular/router';
import {  ActivatedRoute, ParamMap } from '@angular/router';

@Component({
  selector: 'app-new-student-form',
  templateUrl: './new-student-form.component.html',
  styleUrls: ['./new-student-form.component.css']
})
export class NewStudentFormComponent implements OnInit {
  @Input() firstName: string = "";
  @Input() lastName: string = "";
  @Input() DOB: string = "";
  @Input() height: string = "";
  @Input() weight: string = "";
  @Input() street: string = "";
  @Input() city: string = "";
  @Input() state: string = "";
  @Input() zip: string = "";

  public mode = 'Add'; //default mode
  private id: any; //condition ID
  private student: any;

  //initialize the call using StudentService
  constructor(private _myService: StudentService, private router:Router, public route: ActivatedRoute) { }

  ngOnInit() {
    this.route.paramMap.subscribe((paramMap: ParamMap ) => {
        if (paramMap.has('_id')){
            this.mode = 'Edit'; /*request had a parameter _id */ 
            this.id = paramMap.get('_id');

             //request student info based on the id
            this._myService.getStudent(this.id).subscribe(
                data => { 
                    //read data and assign to private variable student
                    this.student = data;
                    //populate the firstName and lastName on the page
                    //notice that this is done through the two-way bindings
                    this.firstName = this.student.firstName;
                    this.lastName = this.student.lastName;
                    this.DOB= this.student.DOB;
                    this.height = this.student.height;
                    this.weight = this.student.weight;
                    this.street= this.student.street;
                    this.city = this.student.city;
                    this.state = this.student.state;
                    this.zip= this.student.zip;
                },
                err => console.error(err),
                () => console.log('finished loading')
            );
        } 
        else {
            this.mode = 'Add';
            this.id = null; 
        }
    });
  } 

  onSubmit(){
    console.log("You submitted: " + this.firstName + " " + this.lastName + " " + this.DOB + " " + this.height + "" +
    this.weight + " " + this.street + " " + this.city + " " + this.state + " " + this. zip);
    if (this.mode == 'Add')
      this._myService.addStudent(this.firstName, this.lastName, this.DOB, this.height, this.weight, this.street, this.city, this.state, this. zip);
    if (this.mode == 'Edit')
      this._myService.updateStudent(this.id, this.firstName,this.lastName, this.DOB, this.height, this.weight, 
        this.street, this.city, this.state, this. zip);
    this.router.navigate(['/listStudents']);
  }

}

