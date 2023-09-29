import { FuzzyProfileDto } from "../../../../modules/fuzzy/dtos/fuzzy-profile-dto";

export class FetchFuzzyProfilesResponseDto {
    fuzzyProfilesNames: string[];
    fuzzyProfilesMap: Record<string, FuzzyProfileDto>;

    constructor(obj: {
        fuzzyProfilesNames: string[];
        fuzzyProfilesMap: Record<string, FuzzyProfileDto>;
    }) {
        this.fuzzyProfilesNames = obj.fuzzyProfilesNames;
        this.fuzzyProfilesMap = obj.fuzzyProfilesMap;
    }

    static fromObj(obj: any) {
        if (!obj) {
            throw new Error('obj cannot be null')
        }
        let fuzzyProfilesMap: Record<string, FuzzyProfileDto> = {};
        if (obj.fuzzyProfilesMap) {
            fuzzyProfilesMap = Object.fromEntries(Object.entries(obj.fuzzyProfilesMap).map(([key, obj]) => [key, FuzzyProfileDto.fromObj(obj)]));

        }
        return new FetchFuzzyProfilesResponseDto({
            fuzzyProfilesNames: obj.fuzzyProfilesNames,
            fuzzyProfilesMap: fuzzyProfilesMap
        })
    }
}