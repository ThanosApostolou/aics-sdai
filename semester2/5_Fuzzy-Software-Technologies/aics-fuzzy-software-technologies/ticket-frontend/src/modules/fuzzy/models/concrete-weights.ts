export class ConcreteWeights {
    choice1: number;
    choice2: number;
    choice3: number;
    choice4: number;

    constructor(obj: {
        choice1: number,
        choice2: number,
        choice3: number,
        choice4: number
    }) {
        this.choice1 = obj.choice1;
        this.choice2 = obj.choice2;
        this.choice3 = obj.choice3;
        this.choice4 = obj.choice4;
    }


    static fromObj(obj: any): ConcreteWeights {
        return new ConcreteWeights({
            choice1: obj.choice1,
            choice2: obj.choice2,
            choice3: obj.choice3,
            choice4: obj.choice4
        })
    }

    deepClone(): ConcreteWeights {
        return new ConcreteWeights({
            choice1: this.choice1,
            choice2: this.choice2,
            choice3: this.choice3,
            choice4: this.choice4,
        })
    }

}
