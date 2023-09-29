import { FuzzyVariableDistributionType } from "../fuzzy-constants";

export interface FuzzyVariableI {
    getFuzzyVariableMap: () => Record<string, FuzzyVariableDistributionPart>;
    getFuzzyVariableColorsMap: () => Record<string, string>;

    getName(): string;

    get1stPart: () => FuzzyVariableDistributionPart;
    get2ndPart: () => FuzzyVariableDistributionPart;
    get3rdPart: () => FuzzyVariableDistributionPart;
    get4thPart: () => FuzzyVariableDistributionPart;


    set1stPart: (fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) => void;
    set2ndPart: (fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) => void;
    set3rdPart: (fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) => void;
    set4thPart: (fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) => void;

}

export type FuzzyVariableDistributionPart = FuzzyVariableDistributionPartTriangular | FuzzyVariableDistributionPartTrapezoidal;

export class FuzzyVariableDistributionPartUtils {

    static fuzzyVariableDistributionPartFromObjNullable(obj: any): FuzzyVariableDistributionPart | null {
        if (obj == null) {
            return null;
        }
        return this.fuzzyVariableDistributionPartFromObj(obj);
    }

    static fuzzyVariableDistributionPartFromObj(obj: any): FuzzyVariableDistributionPart {
        if (obj == null) {
            throw new Error('obj cannot be null');
        }
        if (FuzzyVariableDistributionType.TRIANGULAR === obj.type) {
            return FuzzyVariableDistributionPartTriangular.fromObj(obj);
        } else if (FuzzyVariableDistributionType.TRAPEZOIDAL === obj.type) {
            return FuzzyVariableDistributionPartTrapezoidal.fromObj(obj);
        } else {
            throw new Error('wrong FuzzyVariableDistributionType')
        }
    }
}

export class FuzzyVariableDistributionPartTriangular {
    type: FuzzyVariableDistributionType;
    partName: string;
    a: number | null;
    b: number;
    c: number | null;

    constructor(obj: {
        partName: string,
        a: number | null,
        b: number,
        c: number | null,
    }) {
        this.type = FuzzyVariableDistributionType.TRIANGULAR;
        this.partName = obj.partName;
        this.a = obj.a;
        this.b = obj.b;
        this.c = obj.c;
    }

    isTypeTriangular(): this is FuzzyVariableDistributionPartTriangular {
        return true;
    }

    isTypeTrapezoidal(): this is FuzzyVariableDistributionPartTrapezoidal {
        return false;
    }

    static fromObjNullable(obj: any): FuzzyVariableDistributionPartTriangular | null {
        if (obj == null) {
            return null;
        }
        return this.fromObj(obj);
    }

    static fromObj(obj: any): FuzzyVariableDistributionPartTriangular {
        if (obj == null) {
            throw new Error('obj cannot be null');
        }
        return new FuzzyVariableDistributionPartTriangular({
            partName: obj.partName,
            a: obj.a,
            b: obj.b,
            c: obj.c
        })
    }

    deepClone(): FuzzyVariableDistributionPartTriangular {
        return new FuzzyVariableDistributionPartTriangular({
            partName: this.partName,
            a: this.a,
            b: this.b,
            c: this.c
        })
    }
}


export class FuzzyVariableDistributionPartTrapezoidal {
    type: FuzzyVariableDistributionType;
    partName: string;
    a: number | null;
    b: number;
    c: number;
    d: number | null;

    constructor(obj: {
        partName: string,
        a: number | null;
        b: number;
        c: number;
        d: number | null;
    }) {
        this.type = FuzzyVariableDistributionType.TRAPEZOIDAL;
        this.partName = obj.partName;
        this.a = obj.a;
        this.b = obj.b;
        this.c = obj.c;
        this.d = obj.d;
    }

    isTypeTriangular(): this is FuzzyVariableDistributionPartTriangular {
        return false;
    }

    isTypeTrapezoidal(): this is FuzzyVariableDistributionPartTrapezoidal {
        return true;
    }

    static fromObjNullable(obj: any): FuzzyVariableDistributionPartTrapezoidal | null {
        if (obj == null) {
            return null;
        }
        return this.fromObj(obj);
    }

    static fromObj(obj: any): FuzzyVariableDistributionPartTrapezoidal {
        if (obj == null) {
            throw new Error('obj cannot be null');
        }
        return new FuzzyVariableDistributionPartTrapezoidal({
            partName: obj.partName,
            a: obj.a,
            b: obj.b,
            c: obj.c,
            d: obj.d
        })
    }

    deepClone(): FuzzyVariableDistributionPartTrapezoidal {
        return new FuzzyVariableDistributionPartTrapezoidal({
            partName: this.partName,
            a: this.a,
            b: this.b,
            c: this.c,
            d: this.d
        })
    }
}