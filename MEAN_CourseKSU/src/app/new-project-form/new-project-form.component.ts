import { Component, OnInit, Input } from '@angular/core';
import { CapstoneService } from '../capstone.service';
import {Router} from '@angular/router';
import {ActivatedRoute, ParamMap } from '@angular/router';

@Component({
  selector: 'app-new-project-form',
  templateUrl: './new-project-form.component.html',
  styleUrls: ['./new-project-form.component.css']
})
export class NewProjectFormComponent implements OnInit {
  @Input() projectName: string = "";
  @Input() projectCategory: string = "";

  public mode = 'Add'; //default mode
  private id: any; //project ID
  private project: any;

  constructor(private _myService: CapstoneService, private router:Router, public route: ActivatedRoute) { }

  ngOnInit() {
    this.route.paramMap.subscribe((paramMap: ParamMap ) => {
        if (paramMap.has('_id')){
            this.mode = 'Edit'; /*request had a parameter _id */ 
            this.id = paramMap.get('_id');

             //request student info based on the id
            this._myService.getProject(this.id).subscribe(
                data => { 
                    //read data and assign to private variable student
                    this.project = data;
                    //populate the firstName and lastName on the page
                    //notice that this is done through the two-way bindings
                    this.projectName = this.project.projectName;
                    this.projectCategory = this.project.projectCategory;
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
    console.log("You submitted: " + this.projectName + " " + this.projectCategory);
    if (this.mode == 'Add')
      this._myService.addProject(this.projectName ,this.projectCategory);
    if (this.mode == 'Edit')
      this._myService.updateProject(this.id,this.projectName ,this.projectCategory);
    this.router.navigate(['/listProjects']);
  }

}
