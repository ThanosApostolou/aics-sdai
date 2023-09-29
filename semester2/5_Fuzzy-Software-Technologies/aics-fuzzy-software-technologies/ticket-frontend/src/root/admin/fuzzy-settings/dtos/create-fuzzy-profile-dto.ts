
export class CreateFuzzyProfileResponseDto {
    name: string;
    error: string;

    constructor(obj: {
        name: string,
        error: string,
    }) {
        this.name = obj.name;
        this.error = obj.error;
    }

    static fromObj(obj: any) {
        if (!obj) {
            throw new Error('obj cannot be null')
        }
        return new CreateFuzzyProfileResponseDto({
            name: obj.name,
            error: obj.error
        })
    }
}