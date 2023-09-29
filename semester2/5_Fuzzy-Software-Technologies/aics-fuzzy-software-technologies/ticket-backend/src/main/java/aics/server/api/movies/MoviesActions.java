package aics.server.api.movies;

import aics.domain.fuzzy.FuzzySearchService;
import aics.domain.fuzzy.FuzzySearchValidator;
import aics.domain.fuzzy.dtos.FuzzySearchFiltersDto;
import aics.domain.fuzzy.models.FuzzySearchResult;
import aics.domain.movie.MovieService;
import aics.domain.movie.dtos.MovieDto;
import aics.domain.movie.dtos.MovieListItemDto;
import aics.domain.movie.entities.Movie;
import aics.infrastructure.errors.TicketErrorStatus;
import aics.infrastructure.errors.TicketException;
import aics.server.api.movies.dtos.FetchMovieDetailsResponseDto;
import aics.server.api.movies.dtos.FetchMoviesPlayingNowResponseDto;
import io.quarkus.logging.Log;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.apache.commons.collections4.CollectionUtils;

import java.util.ArrayList;
import java.util.List;

@ApplicationScoped
public class MoviesActions {
    @Inject
    MovieService movieService;
    @Inject
    FuzzySearchValidator fuzzySearchValidator;
    @Inject
    FuzzySearchService fuzzySearchService;

    @Transactional(rollbackOn = Exception.class)
    public FetchMoviesPlayingNowResponseDto doFetchMoviesPlayingNow(FuzzySearchFiltersDto fuzzySearchFiltersDto) throws TicketException {
        Log.info("Start MoviesActions.doFetchMoviesPlayingNow");
        List<Movie> moviesPlayingNow = this.movieService.fetchMoviesPlayingNow();

        boolean hasFuzzySearch = fuzzySearchFiltersDto.getChoice1() != null || fuzzySearchFiltersDto.getChoice2() != null || fuzzySearchFiltersDto.getChoice3() != null || fuzzySearchFiltersDto.getChoice4() != null;

        if (hasFuzzySearch) {
            String error = this.fuzzySearchValidator.validateFuzzySearchFilters(fuzzySearchFiltersDto);
            if (error != null) {
                throw new TicketException(new Exception(error), error, TicketErrorStatus.UNPROCESSABLE_ENTITY_422);
            }
            FuzzySearchResult fuzzySearchResult = this.fuzzySearchService.fuzzySearch(moviesPlayingNow, fuzzySearchFiltersDto);
            boolean showTopsisAnalysis = fuzzySearchResult.getFuzzySearchTopsisAnalysisDto().getFuzzyProfileDto().isShowTopsisAnalysis();
            FetchMoviesPlayingNowResponseDto fetchMoviesPlayingNowResponseDto = new FetchMoviesPlayingNowResponseDto(
                    fuzzySearchResult.getMovieDtos(),
                    showTopsisAnalysis ? fuzzySearchResult.getFuzzySearchTopsisAnalysisDto() : null,
                    true,
                    null
            );
            Log.info("End MoviesActions.doFetchMoviesPlayingNow");
            return fetchMoviesPlayingNowResponseDto;
        } else {
            List<MovieListItemDto> movieDtos = CollectionUtils.isNotEmpty(moviesPlayingNow)
                    ? moviesPlayingNow.stream().map(MovieListItemDto::fromMovie).toList()
                    : new ArrayList<>();
            FetchMoviesPlayingNowResponseDto fetchMoviesPlayingNowResponseDto = new FetchMoviesPlayingNowResponseDto(
                    movieDtos,
                    null,
                    false,
                    null
            );
            Log.info("End MoviesActions.doFetchMoviesPlayingNow");
            return fetchMoviesPlayingNowResponseDto;
        }
    }

    @Transactional(rollbackOn = Exception.class)
    public FetchMovieDetailsResponseDto doFetchMovieDetails(Long movieId) throws TicketException {
        Log.info("Start MoviesActions.doFetchMovieDetails");
        FetchMovieDetailsResponseDto fetchMovieDetailsResponseDto = new FetchMovieDetailsResponseDto();
        Movie movie = this.movieService.fetchMovieById(movieId);
        MovieDto movieDto = MovieDto.fromMovie(movie);

        fetchMovieDetailsResponseDto.setMovie(movieDto);
        Log.info("End MoviesActions.doFetchMovieDetails");
        return fetchMovieDetailsResponseDto;
    }
}