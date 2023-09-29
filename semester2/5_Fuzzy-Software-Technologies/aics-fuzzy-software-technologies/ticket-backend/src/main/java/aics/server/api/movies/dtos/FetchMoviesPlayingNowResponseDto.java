package aics.server.api.movies.dtos;

import aics.domain.fuzzy.dtos.FuzzySearchTopsisAnalysisDto;
import aics.domain.movie.dtos.MovieListItemDto;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.List;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
@AllArgsConstructor
public class FetchMoviesPlayingNowResponseDto implements Serializable {
    private List<MovieListItemDto> movies;
    private FuzzySearchTopsisAnalysisDto fuzzySearchTopsisAnalysisDto;
    private boolean fuzzySearch;
    private Serializable error;
}
