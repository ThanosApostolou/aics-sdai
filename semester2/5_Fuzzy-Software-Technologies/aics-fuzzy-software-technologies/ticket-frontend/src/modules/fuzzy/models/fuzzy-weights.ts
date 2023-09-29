import { FuzzyVariableDistributionPart, FuzzyVariableDistributionPartUtils, FuzzyVariableI } from './fuzzy-variable-distribution';

export enum FuzzyWeightsFields {
    LOW_IMPORTANCE = 'LOW_IMPORTANCE',
    AVERAGE_IMPORTANCE = 'AVERAGE_IMPORTANCE',
    HIGH_IMPORTANCE = 'HIGH_IMPORTANCE',
    VERY_HIGH_IMPORTANCE = 'VERY_HIGH_IMPORTANCE',
}

export class FuzzyWeights implements FuzzyVariableI {
    varLowImportance: FuzzyVariableDistributionPart;
    varAverageImportance: FuzzyVariableDistributionPart;
    varHighImportance: FuzzyVariableDistributionPart;
    varVeryHighImportance: FuzzyVariableDistributionPart;

    constructor(obj: {
        varLowImportance: FuzzyVariableDistributionPart,
        varAverageImportance: FuzzyVariableDistributionPart,
        varHighImportance: FuzzyVariableDistributionPart,
        varVeryHighImportance: FuzzyVariableDistributionPart
    }) {
        this.varLowImportance = obj.varLowImportance;
        this.varAverageImportance = obj.varAverageImportance;
        this.varHighImportance = obj.varHighImportance;
        this.varVeryHighImportance = obj.varVeryHighImportance;
    }

    set1stPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varLowImportance = fuzzyVariableDistributionPart;
    }
    set2ndPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varAverageImportance = fuzzyVariableDistributionPart;
    }
    set3rdPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varHighImportance = fuzzyVariableDistributionPart;
    }
    set4thPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varVeryHighImportance = fuzzyVariableDistributionPart;
    }

    getName(): string {
        return "WEIGHTS";
    }

    get1stPart() {
        return this.varLowImportance;
    }

    get2ndPart() {
        return this.varAverageImportance;
    }

    get3rdPart() {
        return this.varHighImportance;
    }

    get4thPart() {
        return this.varVeryHighImportance;
    }

    static fromObj(obj: any): FuzzyWeights {
        return new FuzzyWeights({
            varLowImportance: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varLowImportance),
            varAverageImportance: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varAverageImportance),
            varHighImportance: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varHighImportance),
            varVeryHighImportance: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varVeryHighImportance)
        })
    }

    deepClone(): FuzzyWeights {
        return new FuzzyWeights({
            varLowImportance: this.varLowImportance.deepClone(),
            varAverageImportance: this.varAverageImportance.deepClone(),
            varHighImportance: this.varHighImportance.deepClone(),
            varVeryHighImportance: this.varVeryHighImportance.deepClone(),
        })
    }

    getFuzzyVariableMap(): Record<string, FuzzyVariableDistributionPart> {
        return {
            [FuzzyWeightsFields.LOW_IMPORTANCE]: this.varLowImportance,
            [FuzzyWeightsFields.AVERAGE_IMPORTANCE]: this.varAverageImportance,
            [FuzzyWeightsFields.HIGH_IMPORTANCE]: this.varHighImportance,
            [FuzzyWeightsFields.VERY_HIGH_IMPORTANCE]: this.varVeryHighImportance,
        }
    }

    getFuzzyVariableColorsMap(): Record<string, string> {
        return {
            [FuzzyWeightsFields.LOW_IMPORTANCE]: 'red',
            [FuzzyWeightsFields.AVERAGE_IMPORTANCE]: 'orange',
            [FuzzyWeightsFields.HIGH_IMPORTANCE]: 'yellow',
            [FuzzyWeightsFields.VERY_HIGH_IMPORTANCE]: 'green',
        }
    }

}
