import { TopsisDataTableDto } from './topsis-data-table-dto';

export class FuzzyTopsisInfoDto {
    table1InitialData: TopsisDataTableDto;
    table2FuzzifiedData: TopsisDataTableDto;
    table3FuzzifiedDistributionDataDto: TopsisDataTableDto;
    table4NormalizedDataDto: TopsisDataTableDto;
    table5WeightedDistributionDataDto: TopsisDataTableDto;
    table6TopsisScoreDto: TopsisDataTableDto;

    constructor(obj: {
        table1InitialData: TopsisDataTableDto,
        table2FuzzifiedData: TopsisDataTableDto,
        table3FuzzifiedDistributionDataDto: TopsisDataTableDto,
        table4NormalizedDataDto: TopsisDataTableDto,
        table5WeightedDistributionDataDto: TopsisDataTableDto,
        table6TopsisScoreDto: TopsisDataTableDto,
    }) {
        this.table1InitialData = obj.table1InitialData;
        this.table2FuzzifiedData = obj.table2FuzzifiedData;
        this.table3FuzzifiedDistributionDataDto = obj.table3FuzzifiedDistributionDataDto;
        this.table4NormalizedDataDto = obj.table4NormalizedDataDto;
        this.table5WeightedDistributionDataDto = obj.table5WeightedDistributionDataDto;
        this.table6TopsisScoreDto = obj.table6TopsisScoreDto;
    }

    static fromObj(obj: any): FuzzyTopsisInfoDto {
        if (obj == null) {
            throw new Error('obj cannot be null')
        }
        return new FuzzyTopsisInfoDto({
            table1InitialData: TopsisDataTableDto.fromObj(obj.table1InitialData),
            table2FuzzifiedData: TopsisDataTableDto.fromObj(obj.table2FuzzifiedData),
            table3FuzzifiedDistributionDataDto: TopsisDataTableDto.fromObj(obj.table3FuzzifiedDistributionDataDto),
            table4NormalizedDataDto: TopsisDataTableDto.fromObj(obj.table4NormalizedDataDto),
            table5WeightedDistributionDataDto: TopsisDataTableDto.fromObj(obj.table5WeightedDistributionDataDto),
            table6TopsisScoreDto: TopsisDataTableDto.fromObj(obj.table6TopsisScoreDto),
        });
    }

    static fromObjNullable(obj: any): FuzzyTopsisInfoDto | null {
        if (obj == null) {
            return null;
        }
        return this.fromObj(obj);
    }
}