import { Component, Input, OnInit } from '@angular/core';
import { AccountDetailsDto } from '../../../dtos/account-details-dto';
import { FormControl, FormGroup } from '@angular/forms';

@Component({
    selector: 'app-tab-user-details',
    templateUrl: './tab-user-details.component.html',
    styleUrls: ['./tab-user-details.component.scss']
})
export class TabUserDetailsComponent implements OnInit {
    @Input()
    public accountDetailsForm!: FormGroup;

    constructor() { }

    ngOnInit(): void {
    }

}
