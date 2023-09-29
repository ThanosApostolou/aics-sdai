package aics.domain.fuzzy.models;

import aics.domain.fuzzy.constants.FuzzyVariableDistributionType;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "type", visible = true)
@JsonSubTypes({
        @JsonSubTypes.Type(value = FuzzyVariableDistributionPartTriangular.class, name = FuzzyVariableDistributionType.CONSTANT_NAME_TRIANGULAR),
        @JsonSubTypes.Type(value = FuzzyVariableDistributionPartTrapezoidal.class, name = FuzzyVariableDistributionType.CONSTANT_NAME_TRAPEZOIDAL),
})
public abstract class FuzzyVariableDistributionPart implements Serializable {
    FuzzyVariableDistributionType type;

    abstract public double findFirstValue();

    abstract public double findLastValue();

    abstract public double calculateFuzzyPartMx(double x);

    abstract public FuzzyValue calculateFuzzyPartValue();
}
