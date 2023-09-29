package aics.domain.fuzzy.dtos;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
@AllArgsConstructor
public class FuzzySearchTopsisAnalysisDto implements Serializable {
    private FuzzyProfileDto fuzzyProfileDto;
    private FuzzySearchFiltersDto fuzzySearchFiltersDto;
    private RegularTopsisInfoDto regularTopsisInfoDto;
    private FuzzyTopsisInfoDto fuzzyTopsisInfoDto;
}
