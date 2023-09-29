package aics.domain.fuzzy.models;

import aics.domain.fuzzy.constants.FuzzyVariableDistributionType;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.Objects;

@EqualsAndHashCode(callSuper = true)
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@Accessors(chain = true)
public class FuzzyVariableDistributionPartTriangular extends FuzzyVariableDistributionPart implements Serializable {
    private String partName;
    private Double a;
    private double b;
    private Double c;

    public FuzzyVariableDistributionPartTriangular(String partName, Double a, double b, Double c) {
        this.setType(FuzzyVariableDistributionType.TRIANGULAR);
        this.partName = partName;
        this.a = a;
        this.b = b;
        this.c = c;
    }

    @Override
    public double findFirstValue() {
        return Objects.requireNonNullElse(this.a, this.b);
    }

    @Override
    public double findLastValue() {
        return Objects.requireNonNullElse(this.c, this.b);
    }

    @Override
    public double calculateFuzzyPartMx(double x) {
        double firstValue = this.findFirstValue();
        double lastValue = this.findLastValue();
        if (x < firstValue || x > lastValue) {
            return 0;
        } else if (x == firstValue) {
            return this.a != null ? 0 : 1;
        } else if (x == lastValue) {
            return this.c != null ? 0 : 1;
        }

        if (x < this.b) {
            return (x - firstValue) / (this.b - firstValue);
        } else {
            return (lastValue - x) / (lastValue - this.b);
        }
    }

    @Override
    public FuzzyValue calculateFuzzyPartValue() {
        return new FuzzyValue(
                this.findFirstValue(),
                this.b,
                this.b,
                this.findLastValue()
        );
    }

}
