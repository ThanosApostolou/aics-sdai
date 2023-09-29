package aics.domain.movie.dtos;

import aics.domain.movie.entities.Movie;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.Base64;

@Data
@Accessors(chain = true)
public class MovieListItemDto implements Serializable {
    private Long movieId;
    private String name;
    private String description;
    private String image;
    private String imageName;
    private String imageMimePrefix;
    private int year;
    private double rating;
    private int popularity;
    private int duration;
    private double topsisScore;

    public static MovieListItemDto fromMovie(Movie movie) {
        if (movie == null) {
            return null;
        }
        return new MovieListItemDto()
                .setMovieId(movie.getMovieId())
                .setName(movie.getName())
                .setDescription(movie.getDescription())
                .setImage(Base64.getEncoder().encodeToString(movie.getImage()))
                .setImageName(movie.getImageName())
                .setImageMimePrefix(movie.getImageMimePrefix())
                .setYear(movie.getYear())
                .setRating(movie.getRating())
                .setPopularity(movie.getPopularity())
                .setDuration(movie.getDuration())
                .setTopsisScore(0);
    }


    public static MovieListItemDto fromMovieAndTopsisScore(Movie movie, double topsisScore) {
        if (movie == null) {
            return null;
        }
        return new MovieListItemDto()
                .setMovieId(movie.getMovieId())
                .setName(movie.getName())
                .setDescription(movie.getDescription())
                .setImage(Base64.getEncoder().encodeToString(movie.getImage()))
                .setImageName(movie.getImageName())
                .setImageMimePrefix(movie.getImageMimePrefix())
                .setYear(movie.getYear())
                .setRating(movie.getRating())
                .setPopularity(movie.getPopularity())
                .setDuration(movie.getDuration())
                .setTopsisScore(topsisScore);
    }
}
