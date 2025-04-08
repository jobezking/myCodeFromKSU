import { Component, OnInit, Input,  } from '@angular/core';
import { PatientService } from '../patient.service';
import {  Router} from '@angular/router';
import {  ActivatedRoute, ParamMap } from '@angular/router';

@Component({
  selector: 'app-new-appointment-form',
  templateUrl: './new-appointment-form.component.html',
  styleUrls: ['./new-appointment-form.component.css']
})
export class NewAppointmentFormComponent implements OnInit {
  @Input() firstName: string = "";
  @Input() lastName: string = "";
  @Input() email: string = "";
  @Input() phone: string = "";
  @Input() reason: string = "";
  @Input() date: string = "";
  @Input() doctorName: string = "";
  @Input() comment: string = "";

  public mode = 'Add'; //default mode
  private id: any; //patient ID
  private patient: any;
  
  constructor(private _myService: PatientService, private router:Router, public route: ActivatedRoute) { }
  

  ngOnInit() {
    this.route.paramMap.subscribe((paramMap: ParamMap ) => {
        if (paramMap.has('_id')){
            this.mode = 'Edit'; /*request had a parameter _id */ 
            this.id = paramMap.get('_id');

             //request student info based on the id
            this._myService.getPatient(this.id).subscribe(
                data => { 
                    //read data and assign to private variable student
                    this.patient = data;
                    //populate the firstName and lastName on the page
                    //notice that this is done through the two-way bindings
                    this.firstName = this.patient.firstName;
                    this.lastName = this.patient.lastName;
                    this.email = this.patient.email;
                    this.phone = this.patient.phone;
                    this.reason = this.patient.reason;
                    this.date = this.patient.date;
                    this.doctorName = this.patient.doctorName;
                    this.comment = this.patient.comment;

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
  if (this.mode == 'Add')
    this._myService.addPatients(this.firstName, this.lastName, this.email, this.phone, this.reason, this.date, this.doctorName, this.comment);
  if (this.mode == 'Edit')
    this._myService.updatePatient(this.id, this.firstName, this.lastName, this.email, this.phone, this.reason, this.date, this.doctorName, this.comment);
  this.router.navigate(['/listPatients']);
  // console.log("You submitted: " + this.firstName + " " + this.lastName + " " + this.email + 
  // this.phone + " " + this.reason + " " + this.date + " " + this.doctorName + " " + this.comment);
  // this._myService.addPatients(this.firstName ,this.lastName, this.email, this.phone, this.reason, this.date, this.doctorName, this.comment);
  // this.router.navigate(['/listPatients']);

}
}



    
//   ngOnInit(): void {
//   }



