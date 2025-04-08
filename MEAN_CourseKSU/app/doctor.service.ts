import {Injectable} from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
//we know that response will be in JSON format
const httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
};

@Injectable()
export class DoctorService {

    constructor(private http:HttpClient) {}

    // Uses http.get() to load data 
    getDoctors() {
        return this.http.get('http://localhost:8000/doctors');
    }

    getDoctor(doctorId: string) {
        return this.http.get('http://localhost:8000/doctors/'+ doctorId);
    }
    //Uses http.post() to post data 
addDoctors(firstName: string, middleName: string,  lastName: string,
    dob: string,
    gender: string,
    email: string,
    street: string,
    city: string,
    state: string,
    zip: string,
    specialty: string,
    degree: string,
    instName: string,
    year: string,
    newpatient: string) {
    this.http.post('http://localhost:8000/doctors',{ firstName, middleName, lastName,
    dob,
    gender,
    email,
    street,
    city,
    state,
    zip,
    specialty,
    degree,
    instName,
    year,
    newpatient  })
        .subscribe((responseData) => {
            console.log(responseData);
        }); 
        
    }
    deleteDoctor(doctorId: string) {
        this.http.delete("http://localhost:8000/doctors/" + doctorId)
            .subscribe(() => {
                console.log('Deleted: ' + doctorId);
            });
            location.reload();
    } 
    updateDoctor(doctorId: string,firstName: string, middleName: string,  lastName: string,
        dob: string,
        gender: string,
        email: string,
        street: string,
        city: string,
        state: string,
        zip: string,
        specialty: string,
        degree: string,
        instName: string,
        year: string,
        newpatient: string) {
        //request path http://localhost:8000/doctors/5xbd456xx 
        //first and last names will be send as HTTP body parameters 
        this.http.put("http://localhost:8000/doctors/" + 
        doctorId,{ firstName, middleName, lastName,
            dob,
            gender,
            email,
            street,
            city,
            state,
            zip,
            specialty,
            degree,
            instName,
            year,
            newpatient })
        .subscribe(() => {
            console.log('Updated: ' + doctorId);
        });
    }
   
}