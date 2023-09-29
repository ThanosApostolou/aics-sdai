import { FuzzySearchChoices } from '../fuzzy-constants';

export class FuzzySearchFiltersDto {
    choice1: FuzzySearchChoices;
    choice2: FuzzySearchChoices;
    choice3: FuzzySearchChoices;
    choice4: FuzzySearchChoices;
    yearCostCriteria: boolean;
    durationCostCriteria: boolean;

    constructor(obj: {
        choice1: FuzzySearchChoices,
        choice2: FuzzySearchChoices,
        choice3: FuzzySearchChoices,
        choice4: FuzzySearchChoices,
        yearCostCriteria: boolean,
        durationCostCriteria: boolean,
    }) {
        this.choice1 = obj.choice1;
        this.choice2 = obj.choice2;
        this.choice3 = obj.choice3;
        this.choice4 = obj.choice4;
        this.yearCostCriteria = obj.yearCostCriteria;
        this.durationCostCriteria = obj.durationCostCriteria;
    }

    static fromObj(obj: any): FuzzySearchFiltersDto {
        if (!obj) {
            throw new Error('obj cannot be null');
        }
        return new FuzzySearchFiltersDto({
            choice1: obj.choice1,
            choice2: obj.choice2,
            choice3: obj.choice3,
            choice4: obj.choice4,
            yearCostCriteria: obj.yearCostCriteria,
            durationCostCriteria: obj.durationCostCriteria,
        });
    }

    deepClone(): FuzzySearchFiltersDto {
        return new FuzzySearchFiltersDto({
            choice1: this.choice1,
            choice2: this.choice2,
            choice3: this.choice3,
            choice4: this.choice4,
            yearCostCriteria: this.yearCostCriteria,
            durationCostCriteria: this.durationCostCriteria,
        })
    }
}