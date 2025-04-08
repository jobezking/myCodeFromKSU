import { Component, OnInit, Input } from '@angular/core';
import { DoctorService } from '../doctor.service';
import {Router} from '@angular/router';
import {ActivatedRoute, ParamMap } from '@angular/router';

@Component({
  selector: 'app-new-doctor-form',
  templateUrl: './new-doctor-form.component.html',
  styleUrls: ['./new-doctor-form.component.css']
})
export class NewDoctorFormComponent implements OnInit {
  @Input() firstName: string = "";
  @Input() lastName: string = "";
  @Input() middleName: string = "";
  @Input() dob: string = "";
  @Input() gender: string = "";
  @Input() email: string = "";
  @Input() street: string = "";
  @Input() city: string = "";
  @Input() state: string = "";
  @Input() zip: string = "";
  @Input() degree: string = "";
  @Input() instName: string = "";
  @Input() year: string = "";
  @Input() specialty: string = "";
  @Input() newpatient: string = "";
  

  public mode = 'Add'; //default mode
  private id: any; //doctor ID;
  
//initialize the call using DoctorService 
constructor(private _myService: DoctorService, private router:Router, public route: ActivatedRoute) { }
                    
ngOnInit(){
  this.route.paramMap.subscribe((paramMap: ParamMap ) => {
      if (paramMap.has('_id'))
          { this.mode = 'Edit'; /*request had a parameter _id */ 
          this.id = paramMap.get('_id');}
      else {this.mode = 'Add';
          this.id = null; }
  });
}

onSubmit(){
    console.log("You submitted: " + this.firstName + " " + this.lastName);
    console.log("You submitted: " + this.dob );
    console.log("You submitted: " + this.gender );
    console.log("You submitted: " + this.email );
    console.log("You submitted: " + this.street + " " + this.city+ " " + this.state + " " + this.zip );
    console.log("You submitted: " + this.degree + " "  + this.instName + " " + this.year);
    console.log("You submitted: " +  this.specialty);
    console.log("You submitted: " + this.newpatient );
    if (this.mode == 'Add')
    this._myService.addDoctors(this.firstName ,this.middleName, this.lastName,
      this.dob,
      this.gender,
      this.email,
      this.street,
      this.city,
      this.state,
      this.zip,
      this.degree,
      this.instName,
      this.year,
      this.specialty,
      this.newpatient);
    if (this.mode == 'Edit')
    this._myService.updateDoctor(this.id,this.firstName ,this.middleName, this.lastName,
      this.dob,
      this.gender,
      this.email,
      this.street,
      this.city,
      this.state,
      this.zip,
      this.degree,
      this.instName,
      this.year,
      this.specialty,
      this.newpatient);
                    
      this.router.navigate(['/listDoctors']);
}
}

