package aics.domain.fuzzy.models;

import aics.domain.fuzzy.constants.FuzzyWeightsFields;
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
public class FuzzyWeights implements Serializable, FuzzyVariableI {
    private FuzzyVariableDistributionPart varLowImportance;
    private FuzzyVariableDistributionPart varAverageImportance;
    private FuzzyVariableDistributionPart varHighImportance;
    private FuzzyVariableDistributionPart varVeryHighImportance;

    @Override
    public HashMap<String, FuzzyVariableDistributionPart> getFuzzyVariableMap() {
        HashMap<String, FuzzyVariableDistributionPart> fuzzyVariableMap = new HashMap<>();
        fuzzyVariableMap.put(FuzzyWeightsFields.LOW_IMPORTANCE.name(), this.varLowImportance);
        fuzzyVariableMap.put(FuzzyWeightsFields.AVERAGE_IMPORTANCE.name(), this.varAverageImportance);
        fuzzyVariableMap.put(FuzzyWeightsFields.HIGH_IMPORTANCE.name(), this.varHighImportance);
        fuzzyVariableMap.put(FuzzyWeightsFields.VERY_HIGH_IMPORTANCE.name(), this.varVeryHighImportance);
        return fuzzyVariableMap;
    }

    @Override
    public FuzzyVariableDistributionPart find1stPart() {
        return this.varLowImportance;
    }

    @Override
    public FuzzyVariableDistributionPart find2ndPart() {
        return this.varAverageImportance;
    }

    @Override
    public FuzzyVariableDistributionPart find3rdPart() {
        return this.varHighImportance;
    }

    @Override
    public FuzzyVariableDistributionPart find4thPart() {
        return this.varVeryHighImportance;
    }
}
