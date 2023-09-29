import { FuzzyVariableDistributionPart, FuzzyVariableDistributionPartUtils, FuzzyVariableI } from './fuzzy-variable-distribution';

export enum FuzzyVariableYearFields {
    OLD = 'OLD',
    RECENT = 'RECENT',
    NEW = 'NEW',
    VERY_NEW = 'VERY_NEW',
}

export class FuzzyVariableYear implements FuzzyVariableI {
    varOld: FuzzyVariableDistributionPart;
    varRecent: FuzzyVariableDistributionPart;
    varNew: FuzzyVariableDistributionPart;
    varVeryNew: FuzzyVariableDistributionPart;

    constructor(obj: {
        varOld: FuzzyVariableDistributionPart,
        varRecent: FuzzyVariableDistributionPart,
        varNew: FuzzyVariableDistributionPart,
        varVeryNew: FuzzyVariableDistributionPart
    }) {
        this.varOld = obj.varOld;
        this.varRecent = obj.varRecent;
        this.varNew = obj.varNew;
        this.varVeryNew = obj.varVeryNew;
    }

    set1stPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varOld = fuzzyVariableDistributionPart;
    }
    set2ndPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varRecent = fuzzyVariableDistributionPart;
    }
    set3rdPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varNew = fuzzyVariableDistributionPart;
    }
    set4thPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varVeryNew = fuzzyVariableDistributionPart;
    }

    getName(): string {
        return "YEAR";
    }

    get1stPart() {
        return this.varOld;
    }

    get2ndPart() {
        return this.varRecent;
    }

    get3rdPart() {
        return this.varNew;
    }

    get4thPart() {
        return this.varVeryNew;
    }

    static fromObj(obj: any): FuzzyVariableYear {
        return new FuzzyVariableYear({
            varOld: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varOld),
            varRecent: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varRecent),
            varNew: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varNew),
            varVeryNew: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varVeryNew)
        })
    }

    deepClone(): FuzzyVariableYear {
        return new FuzzyVariableYear({
            varOld: this.varOld.deepClone(),
            varRecent: this.varRecent.deepClone(),
            varNew: this.varNew.deepClone(),
            varVeryNew: this.varVeryNew.deepClone(),
        })
    }

    getFuzzyVariableMap(): Record<string, FuzzyVariableDistributionPart> {
        return {
            [FuzzyVariableYearFields.OLD]: this.varOld,
            [FuzzyVariableYearFields.RECENT]: this.varRecent,
            [FuzzyVariableYearFields.NEW]: this.varNew,
            [FuzzyVariableYearFields.VERY_NEW]: this.varVeryNew,
        }
    }

    getFuzzyVariableColorsMap(): Record<string, string> {
        return {
            [FuzzyVariableYearFields.OLD]: 'red',
            [FuzzyVariableYearFields.RECENT]: 'orange',
            [FuzzyVariableYearFields.NEW]: 'yellow',
            [FuzzyVariableYearFields.VERY_NEW]: 'green',
        }
    }

}