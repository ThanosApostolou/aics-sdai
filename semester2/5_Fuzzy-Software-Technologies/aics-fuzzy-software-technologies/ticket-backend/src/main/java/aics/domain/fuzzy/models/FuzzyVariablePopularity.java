package aics.domain.fuzzy.models;

import aics.domain.fuzzy.constants.FuzzyVariablePopularityFields;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.HashMap;

@Data
@Accessors(chain = true)
@AllArgsConstructor
@NoArgsConstructor
public class FuzzyVariablePopularity implements Serializable, FuzzyVariableI {
    private FuzzyVariableDistributionPart varVeryPopular;
    private FuzzyVariableDistributionPart varPopular;
    private FuzzyVariableDistributionPart varAverage;
    private FuzzyVariableDistributionPart varUnpopular;

    @Override
    public HashMap<String, FuzzyVariableDistributionPart> getFuzzyVariableMap() {
        HashMap<String, FuzzyVariableDistributionPart> fuzzyVariableMap = new HashMap<>();
        fuzzyVariableMap.put(FuzzyVariablePopularityFields.VERY_POPULAR.name(), this.varVeryPopular);
        fuzzyVariableMap.put(FuzzyVariablePopularityFields.POPULAR.name(), this.varPopular);
        fuzzyVariableMap.put(FuzzyVariablePopularityFields.AVERAGE.name(), this.varAverage);
        fuzzyVariableMap.put(FuzzyVariablePopularityFields.UNPOPULAR.name(), this.varUnpopular);
        return fuzzyVariableMap;
    }

    @Override
    public FuzzyVariableDistributionPart find1stPart() {
        return this.varVeryPopular;
    }

    @Override
    public FuzzyVariableDistributionPart find2ndPart() {
        return this.varPopular;
    }

    @Override
    public FuzzyVariableDistributionPart find3rdPart() {
        return this.varAverage;
    }

    @Override
    public FuzzyVariableDistributionPart find4thPart() {
        return this.varUnpopular;
    }
}
