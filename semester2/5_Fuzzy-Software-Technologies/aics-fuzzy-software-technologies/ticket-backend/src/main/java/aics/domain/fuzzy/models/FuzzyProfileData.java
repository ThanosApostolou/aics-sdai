package aics.domain.fuzzy.models;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
public class FuzzyProfileData implements Serializable {
    private FuzzyVariableYear fuzzyVariableYear;
    private FuzzyVariableRating fuzzyVariableRating;
    private FuzzyVariablePopularity fuzzyVariablePopularity;
    private FuzzyVariableDuration fuzzyVariableDuration;
    private FuzzyWeights fuzzyWeights;
    private ConcreteWeights concreteWeights;
}
