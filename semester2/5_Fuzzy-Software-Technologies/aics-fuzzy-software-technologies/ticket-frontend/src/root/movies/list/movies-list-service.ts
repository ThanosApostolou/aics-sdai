import { GlobalState } from '../../../modules/core/global-state';
import { FuzzySearchFiltersDto } from '../../../modules/fuzzy/dtos/fuzzy-search-filters-dto';
import { FetchMoviesPlayingNowResponseDto } from './dtos/fetch-movies-playing-now-dto';

export class MoviesListService {

    static async fetchMoviesPlayingNow(fuzzySearchFiltersDto: FuzzySearchFiltersDto | null): Promise<FetchMoviesPlayingNowResponseDto> {
        const apiConsumer = GlobalState.instance.apiConsumer;
        const fetchEventsListUrl = '/movies/movies-playing-now'

        const response = await apiConsumer.get(fetchEventsListUrl, {
            params: fuzzySearchFiltersDto != null ? fuzzySearchFiltersDto : undefined
        });
        const fetchMoviesPlayingNowResponseDto: FetchMoviesPlayingNowResponseDto | null = FetchMoviesPlayingNowResponseDto.fromObj(response.data)
        if (!fetchMoviesPlayingNowResponseDto) {
            throw new Error('fetchMoviesPlayingNowResponseDto was null');
        }
        return fetchMoviesPlayingNowResponseDto;
    }

}