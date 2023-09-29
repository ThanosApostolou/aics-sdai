package aics.domain.fuzzy.models;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public interface FuzzyVariableI {
    public static FuzzyValue fuzzyValueMx(FuzzyVariableI fuzzyVariable, double x) {
        final double a = fuzzyVariable.find1stPart().calculateFuzzyPartMx(x);
        final double b = fuzzyVariable.find2ndPart().calculateFuzzyPartMx(x);
        final double c = fuzzyVariable.find3rdPart().calculateFuzzyPartMx(x);
        final double d = fuzzyVariable.find4thPart().calculateFuzzyPartMx(x);
        return new FuzzyValue(a, b, c, d);
    }

    public static FuzzyValue fuzzyValueFromVariableAndValueMx(FuzzyVariableI fuzzyVariable, FuzzyValue fuzzyValueMx) {
        double mxsSum = 0;
        final List<Double> mxs = new ArrayList<>();
        final List<FuzzyVariableDistributionPart> fuzzyVariableDistributionParts = new ArrayList<>();
        if (fuzzyValueMx.a() > 0) {
            mxsSum += fuzzyValueMx.a();
            mxs.add(fuzzyValueMx.a());
            fuzzyVariableDistributionParts.add(fuzzyVariable.find1stPart());
        }
        if (fuzzyValueMx.b() > 0) {
            mxsSum += fuzzyValueMx.b();
            mxs.add(fuzzyValueMx.b());
            fuzzyVariableDistributionParts.add(fuzzyVariable.find2ndPart());
        }
        if (fuzzyValueMx.c() > 0) {
            mxsSum += fuzzyValueMx.c();
            mxs.add(fuzzyValueMx.c());
            fuzzyVariableDistributionParts.add(fuzzyVariable.find3rdPart());
        }
        if (fuzzyValueMx.d() > 0) {
            mxsSum += fuzzyValueMx.d();
            mxs.add(fuzzyValueMx.d());
            fuzzyVariableDistributionParts.add(fuzzyVariable.find4thPart());
        }
        if (mxs.size() == 0) {
            return new FuzzyValue(0, 0, 0, 0);
        }
        double finalA = 0;
        double finalB = 0;
        double finalC = 0;
        double finalD = 0;
        for (int i = 0; i < mxs.size(); i++) {
            double weightedMx = mxs.get(i) / mxsSum;
            FuzzyValue fuzzyValue = fuzzyVariableDistributionParts.get(i).calculateFuzzyPartValue();
            finalA += weightedMx * fuzzyValue.a();
            finalB += weightedMx * fuzzyValue.b();
            finalC += weightedMx * fuzzyValue.c();
            finalD += weightedMx * fuzzyValue.d();
        }
        return new FuzzyValue(
                finalA,
                finalB,
                finalC,
                finalD
        );
    }

    HashMap<String, FuzzyVariableDistributionPart> getFuzzyVariableMap();


    FuzzyVariableDistributionPart find1stPart();

    FuzzyVariableDistributionPart find2ndPart();

    FuzzyVariableDistributionPart find3rdPart();

    FuzzyVariableDistributionPart find4thPart();
}
