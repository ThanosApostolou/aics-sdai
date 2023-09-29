package aics.domain.movie;

import aics.domain.movie.dtos.MovieDto;
import aics.domain.provider.ProviderRepository;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.apache.commons.lang3.StringUtils;

@ApplicationScoped
public class MovieValidator {
    @Inject
    ProviderRepository providerRepository;

    public String validateForCreateMovie(MovieDto movieDto) {
        final String error = this.validateMandatory(movieDto);
        if (StringUtils.isNotEmpty(error)) {
            return error;
        }
        if (movieDto.getMovieId() != null) {
            return "movieDto.getMovieId() should be null";
        }

        return null;
    }

    public String validateForUpdateMovie(MovieDto movieDto) {
        final String error = this.validateMandatory(movieDto);
        if (StringUtils.isNotEmpty(error)) {
            return error;
        }
        if (movieDto.getMovieId() == null) {
            return "movieDto.getMovieId() was null";
        }

        return null;
    }

    private String validateMandatory(MovieDto movieDto) {
        if (movieDto == null) {
            return "movieModel was null";
        }
        if (StringUtils.isEmpty(movieDto.getName())) {
            return "movieModel.getName() was empty";
        }
        if (StringUtils.isEmpty(movieDto.getImage())) {
            return "movieModel.getImage() was empty";
        }
        if (StringUtils.isEmpty(movieDto.getImageName())) {
            return "movieModel.getImageName() was empty";
        }
        if (StringUtils.isEmpty(movieDto.getImageMimePrefix())) {
            return "movieModel.getImageMimePrefix() was empty";
        }
        if (StringUtils.isEmpty(movieDto.getImageName())) {
            return "movieModel.getImageName() was empty";
        }
        if (StringUtils.isEmpty(movieDto.getDirectors())) {
            return "movieModel.getDirectors() was empty";
        }
        if (StringUtils.isEmpty(movieDto.getScript())) {
            return "movieModel.getScript() was empty";
        }
        if (StringUtils.isEmpty(movieDto.getActors())) {
            return "movieModel.getActors() was empty";
        }
        if (StringUtils.isEmpty(movieDto.getAppropriateness())) {
            return "movieModel.getAppropriateness() was empty";
        }
        if (movieDto.getDuration() <= 0) {
            return "movieModel.getDuration() was not positive";
        }
        if (StringUtils.isEmpty(movieDto.getTrailerSrcUrl())) {
            return "movieModel.getTrailerSrcUrl() was empty";
        }
        if (movieDto.getRating() < 1) {
            return "movieModel.getRating() was below 1";
        }
        if (movieDto.getRating() > 10) {
            return "movieModel.getRating() was above 10";
        }
        if (movieDto.getPopularity() < 1) {
            return "movieModel.getPopularity() was below 1";
        }
        return null;
    }

}