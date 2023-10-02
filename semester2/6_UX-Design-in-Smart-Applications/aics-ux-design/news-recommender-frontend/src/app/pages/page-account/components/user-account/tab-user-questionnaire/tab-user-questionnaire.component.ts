import { Component, EventEmitter, HostListener, Input, OnInit, Output } from '@angular/core';
import { FormGroup } from '@angular/forms';
import { CategoriesEnum } from 'src/app/modules/core/enums/categories-enum';

@Component({
    selector: 'app-tab-user-questionnaire',
    templateUrl: './tab-user-questionnaire.component.html',
    styleUrls: ['./tab-user-questionnaire.component.scss']
})
export class TabUserQuestionnaireComponent implements OnInit {
    @Input()
    public accountDetailsForm!: FormGroup;
    @Input()
    public questionnaireFormGroup!: FormGroup

    @Output()
    private saveClicked = new EventEmitter<void>();

    CategoriesEnum = CategoriesEnum;
    allCategories = Object.values(CategoriesEnum);


    rowHeight: number = 1;
    currentWindowWidth: number = 0;
    cols: number = 4;

    constructor() { }

    ngOnInit() {
        this.currentWindowWidth = window.innerWidth;
        this.rowHeight = (window.innerWidth <= 600) ? 1 : 6;
        this.handleSize(this.currentWindowWidth);
    }

    handleSize(currentWindowWidth: number) {
        if (currentWindowWidth <= 600) {
            this.rowHeight = 4
            this.cols = 1;
        } else if (currentWindowWidth <= 800) {
            this.rowHeight = 3
            this.cols = 2;
        } else if (currentWindowWidth <= 1000) {
            this.rowHeight = 4
            this.cols = 2;
        } else {
            this.rowHeight = 1
            this.cols = 4;
        }
    }

    onSaveClicked() {
        this.saveClicked.emit();
    }

    @HostListener('window:resize')
    onResize() {
        this.currentWindowWidth = window.innerWidth;
        this.handleSize(this.currentWindowWidth);
    }

    isFilledChanged(event: any) {
        const checked = event.checked;
        if (checked) {
            this.questionnaireFormGroup.enable()
        } else {
            this.questionnaireFormGroup.disable()
            this.questionnaireFormGroup.get("is_filled")?.enable()

        }

    }
}
