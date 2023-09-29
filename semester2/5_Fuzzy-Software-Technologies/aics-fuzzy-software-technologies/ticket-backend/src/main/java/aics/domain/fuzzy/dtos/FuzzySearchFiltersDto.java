package aics.domain.fuzzy.dtos;

import aics.domain.fuzzy.constants.FuzzySearchChoices;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
@AllArgsConstructor
public class FuzzySearchFiltersDto implements Serializable {
    private FuzzySearchChoices choice1;
    private FuzzySearchChoices choice2;
    private FuzzySearchChoices choice3;
    private FuzzySearchChoices choice4;
    private boolean yearCostCriteria;
    private boolean durationCostCriteria;
}
