import { FuzzyVariableDistributionPart, FuzzyVariableDistributionPartUtils, FuzzyVariableI } from './fuzzy-variable-distribution';

export enum FuzzyVariableDurationFields {
    SMALL = 'SMALL',
    AVERAGE = 'AVERAGE',
    BIG = 'BIG',
    HUGE = 'HUGE',
}

export class FuzzyVariableDuration implements FuzzyVariableI {
    varSmall: FuzzyVariableDistributionPart;
    varAverage: FuzzyVariableDistributionPart;
    varBig: FuzzyVariableDistributionPart;
    varHuge: FuzzyVariableDistributionPart;

    constructor(obj: {
        varSmall: FuzzyVariableDistributionPart,
        varAverage: FuzzyVariableDistributionPart,
        varBig: FuzzyVariableDistributionPart,
        varHuge: FuzzyVariableDistributionPart
    }) {
        this.varSmall = obj.varSmall;
        this.varAverage = obj.varAverage;
        this.varBig = obj.varBig;
        this.varHuge = obj.varHuge;
    }

    set1stPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varSmall = fuzzyVariableDistributionPart;
    }
    set2ndPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varAverage = fuzzyVariableDistributionPart;
    }
    set3rdPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varBig = fuzzyVariableDistributionPart;
    }
    set4thPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varHuge = fuzzyVariableDistributionPart;
    }

    getName(): string {
        return "DURATION";
    }

    get1stPart() {
        return this.varSmall;
    }

    get2ndPart() {
        return this.varAverage;
    }

    get3rdPart() {
        return this.varBig;
    }

    get4thPart() {
        return this.varHuge;
    }

    static fromObj(obj: any): FuzzyVariableDuration {
        return new FuzzyVariableDuration({
            varSmall: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varSmall),
            varAverage: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varAverage),
            varBig: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varBig),
            varHuge: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varHuge)
        })
    }

    deepClone(): FuzzyVariableDuration {
        return new FuzzyVariableDuration({
            varSmall: this.varSmall.deepClone(),
            varAverage: this.varAverage.deepClone(),
            varBig: this.varBig.deepClone(),
            varHuge: this.varHuge.deepClone(),
        })
    }

    getFuzzyVariableMap(): Record<string, FuzzyVariableDistributionPart> {
        return {
            [FuzzyVariableDurationFields.SMALL]: this.varSmall,
            [FuzzyVariableDurationFields.AVERAGE]: this.varAverage,
            [FuzzyVariableDurationFields.BIG]: this.varBig,
            [FuzzyVariableDurationFields.HUGE]: this.varHuge,
        }
    }

    getFuzzyVariableColorsMap(): Record<string, string> {
        return {
            [FuzzyVariableDurationFields.SMALL]: 'red',
            [FuzzyVariableDurationFields.AVERAGE]: 'orange',
            [FuzzyVariableDurationFields.BIG]: 'yellow',
            [FuzzyVariableDurationFields.HUGE]: 'green',
        }
    }


}
