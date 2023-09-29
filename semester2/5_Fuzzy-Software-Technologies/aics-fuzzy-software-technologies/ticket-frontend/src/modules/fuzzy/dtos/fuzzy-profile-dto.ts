import { FuzzyProfileData } from "../models/fuzzy-profile-data";

export class FuzzyProfileDto {
    fuzzyProfileId: number | null;
    name: string;
    fuzzyProfileData: FuzzyProfileData;
    showTopsisAnalysis: boolean;
    active: boolean;
    useFuzzyTopsis: boolean;

    constructor(obj: {
        fuzzyProfileId: number | null,
        name: string,
        fuzzyProfileData: FuzzyProfileData,
        showTopsisAnalysis: boolean,
        active: boolean,
        useFuzzyTopsis: boolean,
    }) {
        this.fuzzyProfileId = obj.fuzzyProfileId;
        this.name = obj.name;
        this.fuzzyProfileData = obj.fuzzyProfileData;
        this.showTopsisAnalysis = obj.showTopsisAnalysis;
        this.active = obj.active;
        this.useFuzzyTopsis = obj.useFuzzyTopsis;
    }

    static fromObj(obj: any): FuzzyProfileDto {
        if (!obj) {
            throw new Error('obj cannot be null');
        }
        return new FuzzyProfileDto({
            fuzzyProfileId: obj.fuzzyProfileId,
            name: obj.name,
            fuzzyProfileData: FuzzyProfileData.fromObj(obj.fuzzyProfileData),
            showTopsisAnalysis: obj.showTopsisAnalysis,
            active: obj.active,
            useFuzzyTopsis: obj.useFuzzyTopsis,
        });
    }

    deepClone(): FuzzyProfileDto {
        return new FuzzyProfileDto({
            fuzzyProfileId: this.fuzzyProfileId,
            name: this.name,
            fuzzyProfileData: this.fuzzyProfileData.deepClone(),
            showTopsisAnalysis: this.showTopsisAnalysis,
            active: this.active,
            useFuzzyTopsis: this.useFuzzyTopsis,
        })
    }
}