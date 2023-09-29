package aics.domain.fuzzy.models;

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
public class FuzzySearchResult implements Serializable {
    private List<MovieListItemDto> movieDtos;
    private FuzzySearchTopsisAnalysisDto fuzzySearchTopsisAnalysisDto;
}
