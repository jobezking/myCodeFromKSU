import {Injectable} from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
//we know that response will be in JSON format
const httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
};

@Injectable()
export class CapstoneService {

    constructor(private http:HttpClient) {}

    // Uses http.get() to load data
    getProjects() {
        return this.http.get('http://localhost:8000/capstone');
    }

   // Uses http.get() to request data based on project id
    getProject(capstoneprojectId: string) {
    return this.http.get('http://localhost:8000/capstone/'+ capstoneprojectId);
    }

    //Uses http.post() to post data
    addProject(projectName: string, projectCategory: string) {
    this.http.post('http://localhost:8000/capstone',{ projectName, projectCategory })
        .subscribe((responseData) => {
            console.log(responseData);
        });
        //location.reload();
    }

    deleteProject(projectName: string) {
        this.http.delete("http://localhost:8000/capstone/" + projectName)
            .subscribe(() => {
                console.log('Deleted: ' + projectName);
            });
            location.reload();
    }

    updateProject(projectId: string,projectName: string, projectCategory: string) {
        //request path http://localhost:8000/students/5xbd456xx
        //first and last names will be send as HTTP body parameters
        this.http.put("http://localhost:8000/capstone/" +
        projectId,{ projectName, projectCategory })
        .subscribe(() => {
            console.log('Updated: ' + projectId);
        });
        //location.reload();
    }
 //
}
