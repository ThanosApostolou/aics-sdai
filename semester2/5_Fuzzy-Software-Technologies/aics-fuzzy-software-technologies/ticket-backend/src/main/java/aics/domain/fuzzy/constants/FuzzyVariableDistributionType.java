package aics.domain.fuzzy.constants;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum FuzzyVariableDistributionType {
    TRIANGULAR,
    TRAPEZOIDAL;

    public static final String CONSTANT_NAME_TRIANGULAR = "TRIANGULAR";
    public static final String CONSTANT_NAME_TRAPEZOIDAL = "TRAPEZOIDAL";
}
