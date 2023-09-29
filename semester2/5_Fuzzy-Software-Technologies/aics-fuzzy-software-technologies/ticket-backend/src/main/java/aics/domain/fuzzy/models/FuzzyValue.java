package aics.domain.fuzzy.models;

import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

@Accessors(chain = true)
public record FuzzyValue(double a, double b, double c, double d) implements Serializable {

    public static double calculateDistance(FuzzyValue fuzzyValue1, FuzzyValue fuzzyValue2) {
        final double roundFactor = 1000.0;
        final double distance = Math.sqrt((1 / 4.0) * (Math.pow(fuzzyValue1.a - fuzzyValue2.a, 2)
                + Math.pow(fuzzyValue1.b - fuzzyValue2.b, 2)
                + Math.pow(fuzzyValue1.c - fuzzyValue2.c, 2)
                + Math.pow(fuzzyValue1.d - fuzzyValue2.d, 2)
        ));
        return Math.round(distance * roundFactor) / roundFactor;
    }

    public static FuzzyValue multiply(FuzzyValue fuzzyValue1, FuzzyValue fuzzyValue2) {
        final double roundFactor = 1000.0;
        return new FuzzyValue(
                Math.round(fuzzyValue1.a * fuzzyValue2.a * roundFactor) / roundFactor,
                Math.round(fuzzyValue1.b * fuzzyValue2.b * roundFactor) / roundFactor,
                Math.round(fuzzyValue1.c * fuzzyValue2.c * roundFactor) / roundFactor,
                Math.round(fuzzyValue1.d * fuzzyValue2.d * roundFactor) / roundFactor
        );
    }

    public double max() {
        return Collections.max(List.of(this.a, this.b, this.c, this.d));
    }

    public double min() {
        return Collections.min(List.of(this.a, this.b, this.c, this.d));
    }

    public String toFormattedString() {
        final double roundFactor = 1000.0;
        final double newA = Math.round(this.a * roundFactor) / roundFactor;
        return "(%s, %s, %s, %s)".formatted(
                String.valueOf(Math.round(this.a * roundFactor) / roundFactor),
                String.valueOf(Math.round(this.b * roundFactor) / roundFactor),
                String.valueOf(Math.round(this.c * roundFactor) / roundFactor),
                String.valueOf(Math.round(this.d * roundFactor) / roundFactor)
        );
    }
}
