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
public class FuzzyVariableDistributionPartTrapezoidal extends FuzzyVariableDistributionPart implements Serializable {
    private String partName;
    private Double a;
    private double b;
    private double c;
    private Double d;

    public FuzzyVariableDistributionPartTrapezoidal(String partName, Double a, double b, double c, Double d) {
        this.setType(FuzzyVariableDistributionType.TRAPEZOIDAL);
        this.partName = partName;
        this.a = a;
        this.b = b;
        this.c = c;
        this.d = d;
    }


    @Override
    public double findFirstValue() {
        return Objects.requireNonNullElse(this.a, this.b);
    }

    @Override
    public double findLastValue() {
        return Objects.requireNonNullElse(this.d, this.c);
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
            return this.d != null ? 0 : 1;
        }

        if (x < this.b) {
            return (x - firstValue) / (this.b - firstValue);
        } else if (x <= this.c) {
            return 1;
        } else {
            return (lastValue - x) / (lastValue - this.c);
        }
    }

    @Override
    public FuzzyValue calculateFuzzyPartValue() {
        return new FuzzyValue(
                this.findFirstValue(),
                this.b,
                this.c,
                this.findLastValue()
        );
    }
}
