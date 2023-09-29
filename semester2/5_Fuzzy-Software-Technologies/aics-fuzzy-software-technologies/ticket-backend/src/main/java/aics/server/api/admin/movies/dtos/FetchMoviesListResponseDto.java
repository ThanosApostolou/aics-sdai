package aics.server.api.admin.movies.dtos;

import aics.domain.movie.dtos.MovieListItemDto;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.List;

@Data
@Accessors(chain = true)
public class FetchMoviesListResponseDto implements Serializable {
    private List<MovieListItemDto> movies;
}
