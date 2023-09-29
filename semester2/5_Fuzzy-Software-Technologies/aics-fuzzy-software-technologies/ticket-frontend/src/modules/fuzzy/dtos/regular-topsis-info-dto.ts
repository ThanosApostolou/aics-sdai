import { TopsisDataTableDto } from './topsis-data-table-dto';

export class RegularTopsisInfoDto {
    table1InitialData: TopsisDataTableDto;
    table2NormalizedData: TopsisDataTableDto;
    table3WeightedNormalizedData: TopsisDataTableDto;
    table4TopsisScoreData: TopsisDataTableDto;

    constructor(obj: {
        table1InitialData: TopsisDataTableDto,
        table2NormalizedData: TopsisDataTableDto,
        table3WeightedNormalizedData: TopsisDataTableDto,
        table4TopsisScoreData: TopsisDataTableDto,
    }) {
        this.table1InitialData = obj.table1InitialData;
        this.table2NormalizedData = obj.table2NormalizedData;
        this.table3WeightedNormalizedData = obj.table3WeightedNormalizedData;
        this.table4TopsisScoreData = obj.table4TopsisScoreData;
    }

    static fromObj(obj: any): RegularTopsisInfoDto {
        if (obj == null) {
            throw new Error('obj cannot be null')
        }
        return new RegularTopsisInfoDto({
            table1InitialData: TopsisDataTableDto.fromObj(obj.table1InitialData),
            table2NormalizedData: TopsisDataTableDto.fromObj(obj.table2NormalizedData),
            table3WeightedNormalizedData: TopsisDataTableDto.fromObj(obj.table3WeightedNormalizedData),
            table4TopsisScoreData: TopsisDataTableDto.fromObj(obj.table4TopsisScoreData),
        });
    }

    static fromObjNullable(obj: any): RegularTopsisInfoDto | null {
        if (obj == null) {
            return null;
        }
        return this.fromObj(obj);
    }
}