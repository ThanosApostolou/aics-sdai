import { Component, OnInit } from '@angular/core';
import { PageAccountService } from '../../page-account.service';
import { AccountDetailsDto } from '../../dtos/account-details-dto';
import { FormControl, FormGroup } from '@angular/forms';
import { CategoriesEnum } from 'src/app/modules/core/enums/categories-enum';
import { GlobalStateStoreService } from 'src/app/modules/core/global-state-store.service';

@Component({
    selector: 'app-user-account',
    templateUrl: './user-account.component.html',
    styleUrls: ['./user-account.component.scss']
})
export class UserAccountComponent implements OnInit {
    public isPageLoading: boolean = false;
    public accountDetails: AccountDetailsDto | null = null;

    public accountDetailsForm: FormGroup | null = null;
    questionnaireFormGroup: FormGroup | null = null;
    public errors: string[] = [];

    userDetails = this.globalStateStoreService.userDetails;

    constructor(private pageAccountService: PageAccountService,
        private globalStateStoreService: GlobalStateStoreService) { }

    ngOnInit() {
        this.userDetails = this.globalStateStoreService.userDetails;
        this.loadData();
    }

    private loadData() {
        this.isPageLoading = true;
        this.pageAccountService.fetchAccountDetails().subscribe({
            next: (result) => {
                console.log('result', result);
                this.accountDetails = result;
                this.buildForm(result);
                this.isPageLoading = false;
            },
            error: (error) => {
                this.isPageLoading = false;
            }
        })

    }

    private buildForm(accountDetails: AccountDetailsDto) {
        const questionnaire = accountDetails.user_questionnaire;
        this.userDetails.update((currentValue) => {
            if (currentValue != null) {
                currentValue.favorite_category = questionnaire ? questionnaire.favorite_category : null;
            }
            return currentValue;
        })
        const disabled = questionnaire == null;
        console.log('questionnaire', questionnaire)
        console.log('disabled', disabled)
        this.questionnaireFormGroup = new FormGroup({
            is_filled: new FormControl(questionnaire != null),
            favorite_category: new FormControl({ value: questionnaire ? questionnaire.favorite_category : CategoriesEnum.business, disabled }),
            is_extrovert: new FormControl({ value: questionnaire ? questionnaire.is_extrovert : false, disabled }),
            age: new FormControl({ value: questionnaire ? questionnaire.age : 18, disabled }),
            educational_level: new FormControl({ value: questionnaire ? questionnaire.educational_level : 5, disabled }),
            sports_interest: new FormControl({ value: questionnaire ? questionnaire.sports_interest : 5, disabled }),
            arts_interest: new FormControl({ value: questionnaire ? questionnaire.arts_interest : 5, disabled }),
        });

        this.accountDetailsForm = new FormGroup({
            name: new FormControl(accountDetails.name),
            email: new FormControl(accountDetails.email),
            questionnaire: this.questionnaireFormGroup
        });

    }


    onSaveClicked() {
        const accountDetailsDto = this.getAccountDetailsDtoFromForm(); this.isPageLoading = true;
        this.isPageLoading = true;
        this.errors = [];
        this.pageAccountService.saveAccountDetails(accountDetailsDto).subscribe({
            next: (result) => {
                const errors = result;
                this.errors = errors;
                if (!errors.length) {
                    this.loadData();
                }
                this.isPageLoading = false;
            },
            error: (error) => {
                this.isPageLoading = false;
            }
        })
    }

    private getAccountDetailsDtoFromForm(): AccountDetailsDto {
        const value = this.accountDetailsForm?.getRawValue();
        console.log('value', value)
        const accountDetailsDto = {
            name: value.name,
            email: value.email,
            user_questionnaire: (value.questionnaire == null || !value.questionnaire.is_filled)
                ? null
                : {
                    favorite_category: value.questionnaire.favorite_category,
                    is_extrovert: value.questionnaire.is_extrovert,
                    age: value.questionnaire.age,
                    educational_level: value.questionnaire.educational_level,
                    sports_interest: value.questionnaire.sports_interest,
                    arts_interest: value.questionnaire.arts_interest,
                }
        }
        console.log('accountDetailsDto', accountDetailsDto)
        return accountDetailsDto;
    }

}
