package aics.domain.fuzzy;

import aics.domain.fuzzy.constants.FuzzySearchChoices;
import aics.domain.fuzzy.dtos.FuzzySearchFiltersDto;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.EnumSet;

@ApplicationScoped
public class FuzzySearchValidator {

    public String validateFuzzySearchFilters(FuzzySearchFiltersDto fuzzySearchFiltersDto) {
        if (fuzzySearchFiltersDto == null) {
            return "fuzzySearchFiltersDto was null";
        }
        boolean isAnyNull = fuzzySearchFiltersDto.getChoice1() == null || fuzzySearchFiltersDto.getChoice2() == null || fuzzySearchFiltersDto.getChoice3() == null || fuzzySearchFiltersDto.getChoice4() == null;
        if (isAnyNull) {
            return "Choices cannot be empty";
        }
        EnumSet<FuzzySearchChoices> fuzzySearchChoices = EnumSet.of(fuzzySearchFiltersDto.getChoice1(),
                fuzzySearchFiltersDto.getChoice2(),
                fuzzySearchFiltersDto.getChoice3(),
                fuzzySearchFiltersDto.getChoice4());
        if (fuzzySearchChoices.size() != FuzzySearchChoices.values().length) {
            return "Each Choice must be unique";
        }
        return null;
    }

}