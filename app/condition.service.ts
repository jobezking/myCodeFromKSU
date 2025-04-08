import {Injectable} from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
//we know that response will be in JSON format
const httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
};

@Injectable()
export class ConditionService {

    constructor(private http:HttpClient) {}

    // Uses http.get() to load data
    getConditions() {
        return this.http.get('http://localhost:8000/conditions');
    }
    //Uses http.post() to post data
    addConditions(conditionName: string, conditionNumber: number, conditionText: string) {
        this.http.post('http://localhost:8000/conditions',{ conditionName, conditionNumber, conditionText })
            .subscribe((responseData) => {
                console.log(responseData);
        });
    }

    deleteCondition(conditionId: string) {
        this.http.delete("http://localhost:8000/conditions/" + conditionId)
            .subscribe(() => {
                console.log('Deleted: ' + conditionId);
            });
            location.reload();
    }

    updateCondition(conditionId: string, conditionName: string, conditionNumber: number, conditionText: string) {
        //request path http://localhost:8000/students/5xbd456xx
        //first and last names will be send as HTTP body parameters
        this.http.put("http://localhost:8000/conditions/" +
        conditionId,{ conditionName, conditionNumber, conditionText })
        .subscribe(() => {
            console.log('Updated: ' + conditionId);
        });
    }

    //Uses http.get() to request data based on student id
    getCondition(conditionId: string) {
        return this.http.get('http://localhost:8000/conditions/' + conditionId);
    }
}
