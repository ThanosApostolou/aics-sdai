package aics.server.api.home;

import aics.domain.movie.MovieService;
import aics.domain.movie.dtos.MovieListItemDto;
import aics.domain.movie.entities.Movie;
import aics.infrastructure.auth.AuthService;
import aics.infrastructure.errors.TicketException;
import aics.server.api.home.dtos.FetchMoviesPlayingNowResponseDto;
import io.quarkus.logging.Log;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.apache.commons.collections4.CollectionUtils;

import java.util.ArrayList;
import java.util.List;

@ApplicationScoped
public class HomeActions {
    @Inject
    MovieService movieService;

    @Inject
    AuthService authService;

    @Transactional(rollbackOn = Exception.class)
    public FetchMoviesPlayingNowResponseDto doFetchMoviesPlayingNow() throws TicketException {
        Log.info("Start HomeActions.doFetchMoviesPlayingNow");
        FetchMoviesPlayingNowResponseDto fetchMoviesPlayingNowResponseDto = new FetchMoviesPlayingNowResponseDto();
        List<Movie> moviesPlayingNow = this.movieService.fetchMoviesPlayingNow();
        List<MovieListItemDto> movieDtos = CollectionUtils.isNotEmpty(moviesPlayingNow)
                ? moviesPlayingNow.stream().map(MovieListItemDto::fromMovie).toList()
                : new ArrayList<>();

        fetchMoviesPlayingNowResponseDto.setMovies(movieDtos);
        Log.info("End HomeActions.doFetchMoviesPlayingNow");
        return fetchMoviesPlayingNowResponseDto;
    }
}