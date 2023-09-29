package aics.domain.fuzzy.models;

import aics.domain.fuzzy.constants.FuzzyVariableYearFields;
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
public class FuzzyVariableYear implements Serializable, FuzzyVariableI {
    private FuzzyVariableDistributionPart varOld;
    private FuzzyVariableDistributionPart varRecent;
    private FuzzyVariableDistributionPart varNew;
    private FuzzyVariableDistributionPart varVeryNew;

    @Override
    public HashMap<String, FuzzyVariableDistributionPart> getFuzzyVariableMap() {
        HashMap<String, FuzzyVariableDistributionPart> fuzzyVariableMap = new HashMap<>();
        fuzzyVariableMap.put(FuzzyVariableYearFields.OLD.name(), this.varOld);
        fuzzyVariableMap.put(FuzzyVariableYearFields.RECENT.name(), this.varRecent);
        fuzzyVariableMap.put(FuzzyVariableYearFields.NEW.name(), this.varNew);
        fuzzyVariableMap.put(FuzzyVariableYearFields.VERY_NEW.name(), this.varVeryNew);
        return fuzzyVariableMap;
    }

    @Override
    public FuzzyVariableDistributionPart find1stPart() {
        return this.varOld;
    }

    @Override
    public FuzzyVariableDistributionPart find2ndPart() {
        return this.varRecent;
    }

    @Override
    public FuzzyVariableDistributionPart find3rdPart() {
        return this.varNew;
    }

    @Override
    public FuzzyVariableDistributionPart find4thPart() {
        return this.varVeryNew;
    }
}
