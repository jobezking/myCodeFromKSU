import { Component, OnInit } from '@angular/core';
import { DoctorService } from '../doctor.service';
@Component({
  selector: 'app-list-doctors',
  templateUrl: './list-doctors.component.html',
  styleUrls: ['./list-doctors.component.css']
})
export class ListDoctorsComponent implements OnInit {
//declare variable to hold response and make it public to be accessible from components.html
public doctors: any;
//initialize the call using StudentService 
constructor(private _myService: DoctorService) { }
ngOnInit() {
    this.getDoctors();
}
//method called OnInit
getDoctors() {
    this._myService.getDoctors().subscribe(
        //read data and assign to public variable students
        data => { this.doctors = data},
        err => console.error(err),
        () => console.log('finished loading')
    );
}
onDelete(doctorId: string) {
  this._myService.deleteDoctor(doctorId);
  location.reload();
}

}
