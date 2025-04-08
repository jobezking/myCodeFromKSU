import { Component, OnInit, Input } from '@angular/core';
import { CapstoneService } from '../capstone.service';
import { FormGroup, FormControl, FormBuilder, Validators, FormArray } from '@angular/forms';

@Component({
  selector: 'app-proposal',
  templateUrl: './proposal.component.html',
  styleUrls: ['./proposal.component.css']
})
export class ProposalComponent implements OnInit {

  public Capstoneproject: any;

  ngOnInit() {}
  //method called by OnInit

onSubmit() {
  // TODO: Use EventEmitter with form value
  console.warn(this.proposalForm.value);
}

constructor(private fb: FormBuilder, private _myService: CapstoneService) { }
  
  proposalForm = this.fb.group({
    contactName: ['', Validators.required],
    contactJob: ['', Validators.required],
    contactEmail: ['', Validators.required],
    contactPhone: ['', Validators.required],
    contactCompany: ['', Validators.required],
    contactAddress: this.fb.group({
      street: ['',Validators.required],
      city: ['',Validators.required],
      state: ['',Validators.required],
      zip: ['',Validators.required]
    }),
    contactWebsite: ['', Validators.required],
    title: [''],
    description: [''],
    skills: [''],
    duties: [''],
    benefit: [''],
    provisions: ['']

  });

  onZipChange(){
    let postal_code = this.proposalForm.get('contactAddress').get('zip').value;
    this.proposalForm.patchValue({
      contactAddress: {
        city: this.getCityByZip(postal_code)        
      }
    });
  }
  
   getCityByZip(postcode:string){
     
     let found = list.find( list => list.zip_code == postcode );
     return found.city; 
    }  

  clearProposal(){
    this.proposalForm.patchValue({
      contactName: "",
      contactJob: "",
      contactEmail: "",
      contactPhone: "",
      contactCompany: "",
      contactWebsite: "",
      title: "",
      description: "",
      skills: "",
      duties: "",
      benefit: "",
      provisions: "",
      contactAddress: {
        street: "",
        city: "",
        state: "",
        zip: "" 
      }
    });
  }
}

const list = [
  {
    zip_code: "30144", 
    city: "Kennesaw",
    state: "GA"
  },
  {
    zip_code: "30152", 
    city: "Kennesaw",
    state: "GA"
  },
  {
    zip_code: "30188", 
    city: "Woodstock",
    state: "GA"
  },
  {
    zip_code: "30189", 
    city: "Woodstock",
    state: "GA"
  },
  {
    zip_code: "30060", 
    city: "Marietta",
    state: "GA"
  },
  {
    zip_code: "30061", 
    city: "Marietta",
    state: "GA"
  },
  {
    zip_code: "30062", 
    city: "Marietta",
    state: "GA"
  },
  {
    zip_code: "30063", 
    city: "Marietta",
    state: "GA"
  },
  {
    zip_code: "30064", 
    city: "Marietta",
    state: "GA"
  },
  {
    zip_code: "30065", 
    city: "Marietta",
    state: "GA"
  },
  {
    zip_code: "30066", 
    city: "Marietta",
    state: "GA"
  },
  {
    zip_code: "30067", 
    city: "Marietta",
    state: "GA"
  },
  {
    zip_code: "30068", 
    city: "Marietta",
    state: "GA"
  },
  {
    zip_code: "30069", 
    city: "Marietta",
    state: "GA"
  }  
]
