import { Component, OnInit } from '@angular/core';
import { ConditionService } from '../condition.service';

@Component({
  selector: 'app-list-conditions',
  templateUrl: './list-conditions.component.html',
  styleUrls: ['./list-conditions.component.css']
})
export class ListConditionsComponent implements OnInit {
  //declare variable to hold response and make it public to be accessible from components.html
  public conditions: any;
  //initialize the call using ConditionService 
  constructor(private _myService: ConditionService) { }
  ngOnInit() {
      this.getConditions();
  }
  //method called OnInit
  getConditions() {
      this._myService.getConditions().subscribe(
          //read data and assign to public variable conditions
          data => { this.conditions = data},
          err => console.error(err),
          () => console.log('finished loading')
      );
  }

  onDelete(conditionId: string) {
    this._myService.deleteCondition(conditionId);
    location.reload();
  }

}
