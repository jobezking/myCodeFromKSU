import {Injectable} from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
//we know that response will be in JSON format
const httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
};

@Injectable()
export class StudentService {

    constructor(private http:HttpClient) {}

    // Uses http.get() to load data 
    getStudents() {
        return this.http.get('http://localhost:8000/students');
    }

    getStudent(studentId: string) {
        return this.http.get('http://localhost:8000/students/'+ studentId);
    }

    updateStudent(studentId: string,firstName: string, lastName: string, DOB: string, height: string, 
                        weight: string, street: string, city: string, state: string, zip: string) {
        //request path http://localhost:8000/students/5xbd456xx 
        //first and last names will be send as HTTP body parameters 
        this.http.put("http://localhost:8000/students/" + 
        studentId,{ firstName, lastName, DOB, height, weight, street, state, city, zip })
        .subscribe(() => {
            console.log('Updated: ' + studentId);
    });
}

    deleteStudent(studentId: string) {
        this.http.delete("http://localhost:8000/students/" + studentId)
            .subscribe(() => {
                console.log('Deleted: ' + studentId);
            });
    }
    //Uses http.post() to post data 
    addStudent(firstName: string, lastName: string, DOB: string, height: string, weight: string, street: string, city: string, state: string, zip: string) {
        this.http.post('http://localhost:8000/students',{ firstName, lastName, DOB, height, weight, street, city, state, zip })
            .subscribe((responseData) => {
                console.log(responseData);
        }); 
    }
}
       