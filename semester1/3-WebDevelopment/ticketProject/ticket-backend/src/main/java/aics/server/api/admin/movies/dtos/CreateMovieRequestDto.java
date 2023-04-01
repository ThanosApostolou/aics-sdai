package aics.server.api.admin.movies.dtos;

import aics.domain.movie.dtos.MovieDto;
import lombok.Data;
import lombok.experimental.Accessors;

import javax.ws.rs.core.MediaType;
import java.io.Serializable;

@Data
@Accessors(chain = true)
public class CreateMovieRequestDto implements Serializable {
    private MovieDto movie;
}
