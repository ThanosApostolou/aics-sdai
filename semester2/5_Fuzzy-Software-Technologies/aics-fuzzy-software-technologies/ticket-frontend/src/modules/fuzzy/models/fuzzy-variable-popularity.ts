import { FuzzyVariableDistributionPart, FuzzyVariableDistributionPartUtils, FuzzyVariableI } from './fuzzy-variable-distribution';

export enum FuzzyVariablePopularityFields {
    VERY_POPULAR = 'VERY_POPULAR',
    POPULAR = 'POPULAR',
    AVERAGE = 'AVERAGE',
    UNPOPULAR = 'UNPOPULAR',
}

export class FuzzyVariablePopularity implements FuzzyVariableI {
    varVeryPopular: FuzzyVariableDistributionPart;
    varPopular: FuzzyVariableDistributionPart;
    varAverage: FuzzyVariableDistributionPart;
    varUnpopular: FuzzyVariableDistributionPart;

    constructor(obj: {
        varVeryPopular: FuzzyVariableDistributionPart,
        varPopular: FuzzyVariableDistributionPart,
        varAverage: FuzzyVariableDistributionPart,
        varUnpopular: FuzzyVariableDistributionPart
    }) {
        this.varVeryPopular = obj.varVeryPopular;
        this.varPopular = obj.varPopular;
        this.varAverage = obj.varAverage;
        this.varUnpopular = obj.varUnpopular;
    }


    set1stPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varVeryPopular = fuzzyVariableDistributionPart;
    }
    set2ndPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varPopular = fuzzyVariableDistributionPart;
    }
    set3rdPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varAverage = fuzzyVariableDistributionPart;
    }
    set4thPart(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        this.varUnpopular = fuzzyVariableDistributionPart;
    }

    getName(): string {
        return "POPULARITY";
    }

    get1stPart() {
        return this.varVeryPopular;
    }

    get2ndPart() {
        return this.varPopular;
    }

    get3rdPart() {
        return this.varAverage;
    }

    get4thPart() {
        return this.varUnpopular;
    }

    static fromObj(obj: any): FuzzyVariablePopularity {
        return new FuzzyVariablePopularity({
            varVeryPopular: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varVeryPopular),
            varPopular: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varPopular),
            varAverage: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varAverage),
            varUnpopular: FuzzyVariableDistributionPartUtils.fuzzyVariableDistributionPartFromObj(obj.varUnpopular)
        })
    }

    deepClone(): FuzzyVariablePopularity {
        return new FuzzyVariablePopularity({
            varVeryPopular: this.varVeryPopular.deepClone(),
            varPopular: this.varPopular.deepClone(),
            varAverage: this.varAverage.deepClone(),
            varUnpopular: this.varUnpopular.deepClone(),
        })
    }

    getFuzzyVariableMap(): Record<string, FuzzyVariableDistributionPart> {
        return {
            [FuzzyVariablePopularityFields.VERY_POPULAR]: this.varVeryPopular,
            [FuzzyVariablePopularityFields.POPULAR]: this.varPopular,
            [FuzzyVariablePopularityFields.AVERAGE]: this.varAverage,
            [FuzzyVariablePopularityFields.UNPOPULAR]: this.varUnpopular,
        }
    }


    getFuzzyVariableColorsMap(): Record<string, string> {
        return {
            [FuzzyVariablePopularityFields.VERY_POPULAR]: 'green',
            [FuzzyVariablePopularityFields.POPULAR]: 'yellow',
            [FuzzyVariablePopularityFields.AVERAGE]: 'orange',
            [FuzzyVariablePopularityFields.UNPOPULAR]: 'red',
        }
    }

}
