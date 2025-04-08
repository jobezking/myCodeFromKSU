import { Component, OnInit, Input } from '@angular/core';
import { Router, ActivatedRoute, ParamMap } from '@angular/router';
import { ConditionService } from '../condition.service';

@Component({
  selector: 'app-new-condition-form',
  templateUrl: './new-condition-form.component.html',
  styleUrls: ['./new-condition-form.component.css']
})
export class NewConditionFormComponent implements OnInit {
  @Input() conditionName: string = "";
  @Input() conditionText: string = "";
  @Input() conditionNumber: number = 0;

  public mode = 'Add'; //default mode
  private id: any; //condition ID
  private condition: any;

  //initialize the call using ConditionService 
  constructor(private _myService: ConditionService, private router:Router, public route: ActivatedRoute) { }

  ngOnInit() {
    this.route.paramMap.subscribe((paramMap: ParamMap ) => {
        if (paramMap.has('_id')){
            this.mode = 'Edit'; /*request had a parameter _id */ 
            this.id = paramMap.get('_id');

             //request student info based on the id
            this._myService.getCondition(this.id).subscribe(
                data => { 
                    //read data and assign to private variable student
                    this.condition = data;
                    //populate the firstName and lastName on the page
                    //notice that this is done through the two-way bindings
                    this.conditionName = this.condition.conditionName;
                    this.conditionNumber = this.condition.conditionNumber;
                    this.conditionText = this.condition.conditionText;
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
    console.log("You submitted: " + this.conditionName + " " + this.conditionNumber + " " + this.conditionText);
    if (this.mode == 'Add')
      this._myService.addConditions(this.conditionName, this.conditionNumber ,this.conditionText);
    if (this.mode == 'Edit')
      this._myService.updateCondition(this.id,this.conditionName, this.conditionNumber ,this.conditionText);
    this.router.navigate(['/listConditions']);
  }

}
