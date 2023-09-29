package aics.domain.movie;

import aics.domain.event.EventRepository;
import aics.domain.event.entities.Event;
import aics.domain.event.models.EventFilters;
import aics.domain.movie.dtos.MovieDto;
import aics.domain.movie.entities.Movie;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.apache.commons.lang3.StringUtils;

import java.time.LocalDateTime;
import java.util.*;

@ApplicationScoped
public class MovieService {
    @Inject
    MovieRepository movieRepository;
    @Inject
    EventRepository eventRepository;
    @Inject
    MovieValidator movieValidator;

    public List<Movie> fetchAllMovies() {
        List<Movie> movies = this.movieRepository.findAll().list();

        return movies;
    }

    public Movie fetchMovieById(Long movieId) {
        Movie movie = this.movieRepository.findById(movieId);

        return movie;
    }


    public List<Movie> fetchMoviesPlayingNow() {
        EventFilters eventFilters = new EventFilters()
                .setFromDate(LocalDateTime.now());
        List<Event> eventsPlayingNow = this.eventRepository.findFiltered(eventFilters);
        Set<Long> moviesIds = new HashSet<>();
        List<Movie> moviesPlayingNow = new ArrayList<>();
        for (Event event : eventsPlayingNow) {
            Movie movie = event.getMovie();
            Long movieId = movie.getMovieId();
            if (!moviesIds.contains(movieId)) {
                moviesIds.add(movieId);
                moviesPlayingNow.add(movie);
            }
        }
        moviesPlayingNow.sort(Comparator.comparing(Movie::getName));
        return moviesPlayingNow;
    }


    public String createMovie(MovieDto movieDto) {
        final String error = this.movieValidator.validateForCreateMovie(movieDto);
        if (StringUtils.isNotEmpty(error)) {
            return error;
        }

        Movie newMovie = new Movie()
                .setDescription(movieDto.getDescription())
                .setImage(Base64.getDecoder().decode(movieDto.getImage()))
                .setImageName(movieDto.getImageName())
                .setImageMimePrefix(movieDto.getImageMimePrefix())
                .setName(movieDto.getName())
                .setDirectors(movieDto.getDirectors())
                .setScript(movieDto.getScript())
                .setActors(movieDto.getActors())
                .setAppropriateness(movieDto.getAppropriateness())
                .setDuration(movieDto.getDuration())
                .setTrailerSrcUrl(movieDto.getTrailerSrcUrl())
                .setYear(movieDto.getYear())
                .setRating(movieDto.getRating())
                .setPopularity(movieDto.getPopularity());

        this.movieRepository.persist(newMovie);

        return null;
    }

    public String updateMovie(MovieDto movieDto) {
        final String error = this.movieValidator.validateForUpdateMovie(movieDto);
        if (StringUtils.isNotEmpty(error)) {
            return error;
        }

        Movie movie = this.movieRepository.findById(movieDto.getMovieId());
        if (movie == null) {
            return "couldn't find movie";
        }

        movie.setDescription(movieDto.getDescription())
                .setImage(Base64.getDecoder().decode(movieDto.getImage()))
                .setImageName(movieDto.getImageName())
                .setImageMimePrefix(movieDto.getImageMimePrefix())
                .setName(movieDto.getName())
                .setDirectors(movieDto.getDirectors())
                .setScript(movieDto.getScript())
                .setActors(movieDto.getActors())
                .setAppropriateness(movieDto.getAppropriateness())
                .setDuration(movieDto.getDuration())
                .setTrailerSrcUrl(movieDto.getTrailerSrcUrl())
                .setYear(movieDto.getYear())
                .setRating(movieDto.getRating())
                .setPopularity(movieDto.getPopularity());

        this.movieRepository.persist(movie);

        return null;
    }

    public String deleteMovieById(Long movieId) {
        if (movieId == null) {
            return "movieId was null";
        }
        Movie movie = this.movieRepository.findById(movieId);
        if (movie == null) {
            return "couldn't find movie";
        }
        this.movieRepository.delete(movie);

        return null;
    }
}