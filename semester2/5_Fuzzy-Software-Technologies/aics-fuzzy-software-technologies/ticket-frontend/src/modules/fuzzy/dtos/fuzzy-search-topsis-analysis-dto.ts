import { FuzzyProfileDto } from './fuzzy-profile-dto';
import { FuzzySearchFiltersDto } from './fuzzy-search-filters-dto';
import { FuzzyTopsisInfoDto } from './fuzzy-topsis-info-dto';
import { RegularTopsisInfoDto } from './regular-topsis-info-dto';

export class FuzzySearchTopsisAnalysisDto {
    fuzzyProfileDto: FuzzyProfileDto;
    fuzzySearchFiltersDto: FuzzySearchFiltersDto;
    regularTopsisInfoDto: RegularTopsisInfoDto | null;
    fuzzyTopsisInfoDto: FuzzyTopsisInfoDto | null;

    constructor(obj: {
        fuzzyProfileDto: FuzzyProfileDto,
        fuzzySearchFiltersDto: FuzzySearchFiltersDto,
        regularTopsisInfoDto: RegularTopsisInfoDto | null,
        fuzzyTopsisInfoDto: FuzzyTopsisInfoDto | null,
    }) {
        this.fuzzyProfileDto = obj.fuzzyProfileDto;
        this.fuzzySearchFiltersDto = obj.fuzzySearchFiltersDto;
        this.regularTopsisInfoDto = obj.regularTopsisInfoDto;
        this.fuzzyTopsisInfoDto = obj.fuzzyTopsisInfoDto;
    }

    static fromObj(obj: any): FuzzySearchTopsisAnalysisDto {
        if (obj == null) {
            throw new Error('obj cannot be null')
        }
        return new FuzzySearchTopsisAnalysisDto({
            fuzzyProfileDto: FuzzyProfileDto.fromObj(obj.fuzzyProfileDto),
            fuzzySearchFiltersDto: FuzzySearchFiltersDto.fromObj(obj.fuzzySearchFiltersDto),
            regularTopsisInfoDto: RegularTopsisInfoDto.fromObjNullable(obj.regularTopsisInfoDto),
            fuzzyTopsisInfoDto: FuzzyTopsisInfoDto.fromObjNullable(obj.fuzzyTopsisInfoDto),
        });
    }

    static fromObjNullable(obj: any): FuzzySearchTopsisAnalysisDto | null {
        if (obj == null) {
            return null;
        }
        return this.fromObj(obj);
    }
}