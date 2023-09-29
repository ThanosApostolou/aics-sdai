import { TopsisDataRowDto } from './topsis-data-row-dto';

export class TopsisDataTableDto {
    rows: TopsisDataRowDto[];
    showDpos: boolean;
    showDneg: boolean;
    showScore: boolean;

    constructor(obj: {
        rows: TopsisDataRowDto[],
        showDpos: boolean,
        showDneg: boolean,
        showScore: boolean,
    }) {
        this.rows = obj.rows;
        this.showDpos = obj.showDpos;
        this.showDneg = obj.showDneg;
        this.showScore = obj.showScore;
    }

    static fromObj(obj: any): TopsisDataTableDto {
        return new TopsisDataTableDto({
            rows: obj.rows ? (obj.rows as any[]).map(row => TopsisDataRowDto.fromObj(row)) : [],
            showDpos: obj.showDpos,
            showDneg: obj.showDneg,
            showScore: obj.showScore,
        })
    }

}
