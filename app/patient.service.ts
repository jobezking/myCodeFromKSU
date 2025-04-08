import {Injectable} from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
//we know that response will be in JSON format
const httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
};

@Injectable()
export class PatientService {

    constructor(private http:HttpClient) {}

    // Uses http.get() to load data 
    getPatients() {
        return this.http.get('http://localhost:8000/patients');
    }

    //Uses http.post() to post data 
    addPatients(firstName: string, lastName: string, email: string, phone: string, reason: string, date: string, doctorName: string, comment: string) {
        this.http.post('http://localhost:8000/patients',{ firstName, lastName, email, phone, reason, date, doctorName, comment })
            .subscribe((responseData) => {
                console.log(responseData);
            }); 
          
        }

    deletePatient(patientId: string) {
        this.http.delete("http://localhost:8000/patients/" + patientId)
            .subscribe(() => {
                console.log('Deleted: ' + patientId);
            });
            // location.reload();
}
    updatePatient(patientId: string,firstName: string, lastName: string, email: string, phone: string, reason: string, date: string, doctorName: string, comment: string) {
        //request path http://localhost:8000/students/5xbd456xx 
        //first and last names will be send as HTTP body parameters 
        this.http.put("http://localhost:8000/patients/" + 
        patientId,{ firstName, lastName, email, phone, reason, date, doctorName, comment })
        .subscribe(() => {
            console.log('Updated: ' + patientId);
    });
}
        //Uses http.get() to request data based on student id 
    getPatient(patientId: string) {
        return this.http.get('http://localhost:8000/patients/'+ patientId);
    }
        

          
}

