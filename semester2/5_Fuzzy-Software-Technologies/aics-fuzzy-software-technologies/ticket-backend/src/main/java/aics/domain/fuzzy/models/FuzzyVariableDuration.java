package aics.domain.fuzzy.models;

import aics.domain.fuzzy.constants.FuzzyVariableDurationFields;
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
public class FuzzyVariableDuration implements Serializable, FuzzyVariableI {
    private FuzzyVariableDistributionPart varSmall;
    private FuzzyVariableDistributionPart varAverage;
    private FuzzyVariableDistributionPart varBig;
    private FuzzyVariableDistributionPart varHuge;

    @Override
    public HashMap<String, FuzzyVariableDistributionPart> getFuzzyVariableMap() {
        HashMap<String, FuzzyVariableDistributionPart> fuzzyVariableMap = new HashMap<>();
        fuzzyVariableMap.put(FuzzyVariableDurationFields.SMALL.name(), this.varSmall);
        fuzzyVariableMap.put(FuzzyVariableDurationFields.AVERAGE.name(), this.varAverage);
        fuzzyVariableMap.put(FuzzyVariableDurationFields.BIG.name(), this.varBig);
        fuzzyVariableMap.put(FuzzyVariableDurationFields.HUGE.name(), this.varHuge);
        return fuzzyVariableMap;
    }

    @Override
    public FuzzyVariableDistributionPart find1stPart() {
        return this.varSmall;
    }

    @Override
    public FuzzyVariableDistributionPart find2ndPart() {
        return this.varAverage;
    }

    @Override
    public FuzzyVariableDistributionPart find3rdPart() {
        return this.varBig;
    }

    @Override
    public FuzzyVariableDistributionPart find4thPart() {
        return this.varHuge;
    }
}
