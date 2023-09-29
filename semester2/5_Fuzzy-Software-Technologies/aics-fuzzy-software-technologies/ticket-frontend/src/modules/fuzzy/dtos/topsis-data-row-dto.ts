export class TopsisDataRowDto {
    movieId: number;
    name: string;
    rating: number | string;
    popularity: number | string;
    year: number | string;
    duration: number | string;
    dpos: number | string;
    dneg: number | string;
    score: number | string;

    constructor(obj: {
        movieId: number,
        name: string,
        rating: number | string,
        popularity: number | string,
        year: number | string,
        duration: number | string,
        dpos: number | string,
        dneg: number | string,
        score: number | string,
    }) {
        this.movieId = obj.movieId;
        this.name = obj.name;
        this.rating = obj.rating;
        this.popularity = obj.popularity;
        this.year = obj.year;
        this.duration = obj.duration;
        this.dpos = obj.dpos;
        this.dneg = obj.dneg;
        this.score = obj.score;
    }

    static fromObj(obj: any): TopsisDataRowDto {
        return new TopsisDataRowDto({
            movieId: obj.movieId,
            name: obj.name,
            rating: obj.rating,
            popularity: obj.popularity,
            year: obj.year,
            duration: obj.duration,
            dpos: obj.dpos,
            dneg: obj.dneg,
            score: obj.score,
        })
    }

}
