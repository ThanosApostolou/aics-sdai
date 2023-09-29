package aics.domain.fuzzy;

import aics.domain.fuzzy.constants.*;
import aics.domain.fuzzy.dtos.FuzzyProfileDto;
import aics.domain.fuzzy.models.*;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.apache.commons.lang3.StringUtils;

import java.util.List;

@ApplicationScoped
public class FuzzyProfileValidator {
    @Inject
    FuzzyProfileRepository fuzzyProfileRepository;

    public String validateForCreateFuzzyProfile(FuzzyProfileDto fuzzyProfileDto) {
        final String error = this.validateCommonCreateUpdate(fuzzyProfileDto);
        if (StringUtils.isNotEmpty(error)) {
            return error;
        }
        if (fuzzyProfileDto.getFuzzyProfileId() != null) {
            return "fuzzyProfileDto.getFuzzyProfileId() should be null";
        }

        return null;
    }

    public String validateForUpdateFuzzyProfile(FuzzyProfileDto fuzzyProfileDto) {
        final String error = this.validateCommonCreateUpdate(fuzzyProfileDto);
        if (StringUtils.isNotEmpty(error)) {
            return error;
        }
        if (fuzzyProfileDto.getFuzzyProfileId() == null) {
            return "fuzzyProfileDto.getFuzzyProfileId() should not be null";
        }

        return null;
    }

    private String validateCommonCreateUpdate(FuzzyProfileDto fuzzyProfileDto) {
        if (fuzzyProfileDto == null) {
            return "fuzzyProfileDto was null";
        }
        if (StringUtils.isEmpty(fuzzyProfileDto.getName())) {
            return "fuzzyProfileDto.getName() was empty";
        }

        if (StringUtils.equals(fuzzyProfileDto.getName(), FuzzyConstants.DEFAULT)
                || StringUtils.equals(fuzzyProfileDto.getName(), FuzzyConstants.NEW)
        ) {
            return "fuzzyProfileDto.getName() cannot be DEFAULT or NEW";
        }
        return this.validateFuzzyProfileData(fuzzyProfileDto.getFuzzyProfileData());
    }

    private String validateFuzzyProfileData(FuzzyProfileData fuzzyProfileData) {
        if (fuzzyProfileData == null) {
            return "fuzzyProfileData was null";
        }
        String error = this.validateFuzzyVariableYear(fuzzyProfileData.getFuzzyVariableYear());
        if (error != null) {
            return "FuzzyVariableYear Error: " + error;
        }
        error = this.validateFuzzyVariableDuration(fuzzyProfileData.getFuzzyVariableDuration());
        if (error != null) {
            return "FuzzyVariableDuration Error: " + error;
        }
        error = this.validateFuzzyVariableRating(fuzzyProfileData.getFuzzyVariableRating());
        if (error != null) {
            return "FuzzyVariableRating Error: " + error;
        }
        error = this.validateFuzzyVariablePopularity(fuzzyProfileData.getFuzzyVariablePopularity());
        if (error != null) {
            return "FuzzyVariablePopularity Error: " + error;
        }
        error = this.validateFuzzyWeights(fuzzyProfileData.getFuzzyWeights());
        if (error != null) {
            return "FuzzyWeights Error: " + error;
        }
        error = this.validateConcreteWeights(fuzzyProfileData.getConcreteWeights());
        if (error != null) {
            return "ConcreteWeights Error: " + error;
        }
        return null;
    }

    private String validateFuzzyVariableYear(FuzzyVariableYear fuzzyVariableYear) {
        if (fuzzyVariableYear == null) {
            return "fuzzyVariableYear was null";
        }
        final double minYear = 1900;
        final double maxYear = 2100;
        String error = this.validateFuzzyVariableDistributionPart(fuzzyVariableYear.getVarOld(), FuzzyVariableYearFields.OLD.name(), FuzzyVariablePartPosition.START, minYear, maxYear);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariableYear.getVarRecent(), FuzzyVariableYearFields.RECENT.name(), FuzzyVariablePartPosition.MIDDLE, minYear, maxYear);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariableYear.getVarNew(), FuzzyVariableYearFields.NEW.name(), FuzzyVariablePartPosition.MIDDLE, minYear, maxYear);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariableYear.getVarVeryNew(), FuzzyVariableYearFields.VERY_NEW.name(), FuzzyVariablePartPosition.END, minYear, maxYear);
        if (error != null) {
            return error;
        }
        return this.validateFuzzyVariableDistributionPartsBoundaries(List.of(fuzzyVariableYear.getVarOld(), fuzzyVariableYear.getVarRecent(), fuzzyVariableYear.getVarNew(), fuzzyVariableYear.getVarVeryNew()));
    }

    private String validateFuzzyVariableDuration(FuzzyVariableDuration fuzzyVariableDuration) {
        if (fuzzyVariableDuration == null) {
            return "fuzzyVariableDuration was null";
        }
        final double min = 1;
        final double max = 400;
        String error = this.validateFuzzyVariableDistributionPart(fuzzyVariableDuration.getVarSmall(), FuzzyVariableDurationFields.SMALL.name(), FuzzyVariablePartPosition.START, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariableDuration.getVarAverage(), FuzzyVariableDurationFields.AVERAGE.name(), FuzzyVariablePartPosition.MIDDLE, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariableDuration.getVarBig(), FuzzyVariableDurationFields.BIG.name(), FuzzyVariablePartPosition.MIDDLE, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariableDuration.getVarHuge(), FuzzyVariableDurationFields.HUGE.name(), FuzzyVariablePartPosition.END, min, max);
        if (error != null) {
            return error;
        }
        return this.validateFuzzyVariableDistributionPartsBoundaries(List.of(fuzzyVariableDuration.getVarSmall(), fuzzyVariableDuration.getVarAverage(), fuzzyVariableDuration.getVarBig(), fuzzyVariableDuration.getVarHuge()));
    }


    private String validateFuzzyVariableRating(FuzzyVariableRating fuzzyVariableRating) {
        if (fuzzyVariableRating == null) {
            return "fuzzyVariableRating was null";
        }
        final double min = 1;
        final double max = 10;
        String error = this.validateFuzzyVariableDistributionPart(fuzzyVariableRating.getVarBad(), FuzzyVariableRatingFields.BAD.name(), FuzzyVariablePartPosition.START, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariableRating.getVarAverage(), FuzzyVariableRatingFields.AVERAGE.name(), FuzzyVariablePartPosition.MIDDLE, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariableRating.getVarGood(), FuzzyVariableRatingFields.GOOD.name(), FuzzyVariablePartPosition.MIDDLE, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariableRating.getVarVeryGood(), FuzzyVariableRatingFields.VERY_GOOD.name(), FuzzyVariablePartPosition.END, min, max);
        if (error != null) {
            return error;
        }
        return this.validateFuzzyVariableDistributionPartsBoundaries(List.of(fuzzyVariableRating.getVarBad(), fuzzyVariableRating.getVarAverage(), fuzzyVariableRating.getVarGood(), fuzzyVariableRating.getVarVeryGood()));
    }


    private String validateFuzzyVariablePopularity(FuzzyVariablePopularity fuzzyVariablePopularity) {
        if (fuzzyVariablePopularity == null) {
            return "fuzzyVariablePopularity was null";
        }
        final double min = 1;
        final double max = 1000;
        String error = this.validateFuzzyVariableDistributionPart(fuzzyVariablePopularity.getVarVeryPopular(), FuzzyVariablePopularityFields.VERY_POPULAR.name(), FuzzyVariablePartPosition.START, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariablePopularity.getVarPopular(), FuzzyVariablePopularityFields.POPULAR.name(), FuzzyVariablePartPosition.MIDDLE, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariablePopularity.getVarAverage(), FuzzyVariablePopularityFields.AVERAGE.name(), FuzzyVariablePartPosition.MIDDLE, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyVariablePopularity.getVarUnpopular(), FuzzyVariablePopularityFields.UNPOPULAR.name(), FuzzyVariablePartPosition.END, min, max);
        if (error != null) {
            return error;
        }
        return this.validateFuzzyVariableDistributionPartsBoundaries(List.of(fuzzyVariablePopularity.getVarVeryPopular(), fuzzyVariablePopularity.getVarPopular(), fuzzyVariablePopularity.getVarAverage(), fuzzyVariablePopularity.getVarUnpopular()));
    }

    private String validateFuzzyWeights(FuzzyWeights fuzzyWeights) {
        if (fuzzyWeights == null) {
            return "fuzzyWeights was null";
        }
        final double min = 0.1;
        final double max = 1.0;
        String error = this.validateFuzzyVariableDistributionPart(fuzzyWeights.getVarLowImportance(), FuzzyWeightsFields.LOW_IMPORTANCE.name(), FuzzyVariablePartPosition.START, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyWeights.getVarAverageImportance(), FuzzyWeightsFields.AVERAGE_IMPORTANCE.name(), FuzzyVariablePartPosition.MIDDLE, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyWeights.getVarHighImportance(), FuzzyWeightsFields.HIGH_IMPORTANCE.name(), FuzzyVariablePartPosition.MIDDLE, min, max);
        if (error != null) {
            return error;
        }
        error = this.validateFuzzyVariableDistributionPart(fuzzyWeights.getVarVeryHighImportance(), FuzzyWeightsFields.VERY_HIGH_IMPORTANCE.name(), FuzzyVariablePartPosition.END, min, max);
        if (error != null) {
            return error;
        }
        return this.validateFuzzyVariableDistributionPartsBoundaries(List.of(fuzzyWeights.getVarLowImportance(), fuzzyWeights.getVarAverageImportance(), fuzzyWeights.getVarHighImportance(), fuzzyWeights.getVarVeryHighImportance()));
    }

    private String validateFuzzyVariableDistributionPart(FuzzyVariableDistributionPart fuzzyVariableDistributionPart, String expectedPartName, FuzzyVariablePartPosition fuzzyVariablePartPosition, double minValue, double maxValue) {
        if (fuzzyVariableDistributionPart == null) {
            return "fuzzyVariableDistributionPart was null";
        }
        if (fuzzyVariableDistributionPart instanceof FuzzyVariableDistributionPartTrapezoidal fuzzyVariableDistributionPartTrapezoidal) {
            if (FuzzyVariablePartPosition.START == fuzzyVariablePartPosition) {
                if (fuzzyVariableDistributionPartTrapezoidal.getA() != null) {
                    return "PART %S: fuzzyVariableDistributionPartTrapezoidal.getA() should be null".formatted(fuzzyVariableDistributionPartTrapezoidal.getPartName());
                }
            } else if (FuzzyVariablePartPosition.MIDDLE == fuzzyVariablePartPosition) {
                if (fuzzyVariableDistributionPartTrapezoidal.getA() == null || fuzzyVariableDistributionPartTrapezoidal.getD() == null) {
                    return "PART %S: fuzzyVariableDistributionPartTrapezoidal.getA() and getD() should not be null".formatted(fuzzyVariableDistributionPartTrapezoidal.getPartName());
                }
            } else if (FuzzyVariablePartPosition.END == fuzzyVariablePartPosition) {
                if (fuzzyVariableDistributionPartTrapezoidal.getD() != null) {
                    return "PART %S: fuzzyVariableDistributionPartTrapezoidal.getD() should be null".formatted(fuzzyVariableDistributionPartTrapezoidal.getPartName());
                }
            }
            if (fuzzyVariableDistributionPartTrapezoidal.getA() != null) {
                String error = this.validateFuzzyVariableDistributionPartNumberLimits(fuzzyVariableDistributionPartTrapezoidal.getA(), null, minValue, maxValue);
                if (error != null) {
                    return error;
                }
            }
            String error = this.validateFuzzyVariableDistributionPartNumberLimits(fuzzyVariableDistributionPartTrapezoidal.getB(), fuzzyVariableDistributionPartTrapezoidal.getA(), minValue, maxValue);
            if (error != null) {
                return error;
            }
            error = this.validateFuzzyVariableDistributionPartNumberLimits(fuzzyVariableDistributionPartTrapezoidal.getC(), fuzzyVariableDistributionPartTrapezoidal.getB(), minValue, maxValue);
            if (error != null) {
                return error;
            }
            if (fuzzyVariableDistributionPartTrapezoidal.getD() != null) {
                return this.validateFuzzyVariableDistributionPartNumberLimits(fuzzyVariableDistributionPartTrapezoidal.getD(), fuzzyVariableDistributionPartTrapezoidal.getC(), minValue, maxValue);
            }
        } else if (fuzzyVariableDistributionPart instanceof FuzzyVariableDistributionPartTriangular fuzzyVariableDistributionPartTriangular) {
            if (!StringUtils.equals(expectedPartName, fuzzyVariableDistributionPartTriangular.getPartName())) {
                return "expectedPartName was %s but got %s".formatted(expectedPartName, fuzzyVariableDistributionPartTriangular.getPartName());
            }
            if (FuzzyVariablePartPosition.START == fuzzyVariablePartPosition) {
                if (fuzzyVariableDistributionPartTriangular.getA() != null) {
                    return "PART %S: fuzzyVariableDistributionPartTriangular.getA() should be null".formatted(fuzzyVariableDistributionPartTriangular.getPartName());
                }
            } else if (FuzzyVariablePartPosition.MIDDLE == fuzzyVariablePartPosition) {
                if (fuzzyVariableDistributionPartTriangular.getA() == null || fuzzyVariableDistributionPartTriangular.getC() == null) {
                    return "PART %S: fuzzyVariableDistributionPartTriangular.getA() and getC() should not be null".formatted(fuzzyVariableDistributionPartTriangular.getPartName());
                }
            } else if (FuzzyVariablePartPosition.END == fuzzyVariablePartPosition) {
                if (fuzzyVariableDistributionPartTriangular.getC() != null) {
                    return "PART %S: fuzzyVariableDistributionPartTriangular.getC() should be null".formatted(fuzzyVariableDistributionPartTriangular.getPartName());
                }
            }
            if (fuzzyVariableDistributionPartTriangular.getA() != null) {
                String error = this.validateFuzzyVariableDistributionPartNumberLimits(fuzzyVariableDistributionPartTriangular.getA(), null, minValue, maxValue);
                if (error != null) {
                    return error;
                }
            }
            String error = this.validateFuzzyVariableDistributionPartNumberLimits(fuzzyVariableDistributionPartTriangular.getB(), fuzzyVariableDistributionPartTriangular.getA(), minValue, maxValue);
            if (error != null) {
                return error;
            }
            if (fuzzyVariableDistributionPartTriangular.getC() != null) {
                return this.validateFuzzyVariableDistributionPartNumberLimits(fuzzyVariableDistributionPartTriangular.getC(), fuzzyVariableDistributionPartTriangular.getB(), minValue, maxValue);
            }
        }
        return null;
    }

    private String validateFuzzyVariableDistributionPartNumberLimits(double value, Double previousValue, double minValue, double maxValue) {
        double previousValueOrMin = previousValue != null ? previousValue : minValue - 1;
        if (value <= previousValueOrMin) {
            return "value %s must be higher than previous value %s".formatted(value, previousValueOrMin);
        }
        if (value < minValue) {
            return "value %s must be higher or equal than minValue %s".formatted(value, minValue);
        }
        if (value > maxValue) {
            return "value %s must be less or equal than maxValue %s".formatted(value, maxValue);
        }
        return null;
    }


    private String validateFuzzyVariableDistributionPartsBoundaries(List<FuzzyVariableDistributionPart> fuzzyVariableDistributionParts) {
        int index = 0;
        FuzzyVariableDistributionPart previousFuzzyVariableDistributionPart = fuzzyVariableDistributionParts.get(0);
        for (FuzzyVariableDistributionPart fuzzyVariableDistributionPart : fuzzyVariableDistributionParts) {
            if (index > 0) {
                String partName = "";
                if (fuzzyVariableDistributionPart instanceof FuzzyVariableDistributionPartTriangular fuzzyVariableDistributionPartTriangular) {
                    partName = fuzzyVariableDistributionPartTriangular.getPartName();
                } else if (fuzzyVariableDistributionPart instanceof FuzzyVariableDistributionPartTrapezoidal fuzzyVariableDistributionPartTrapezoidal) {
                    partName = fuzzyVariableDistributionPartTrapezoidal.getPartName();
                }
                if (fuzzyVariableDistributionPart.findFirstValue() <= previousFuzzyVariableDistributionPart.findFirstValue()
                        || fuzzyVariableDistributionPart.findFirstValue() >= previousFuzzyVariableDistributionPart.findLastValue()
                ) {
                    return "PART %s: firstValue %s must be between %s and %s".formatted(partName, fuzzyVariableDistributionPart.findFirstValue(), previousFuzzyVariableDistributionPart.findFirstValue(), previousFuzzyVariableDistributionPart.findLastValue());
                }
                if (fuzzyVariableDistributionPart.findLastValue() <= previousFuzzyVariableDistributionPart.findLastValue()) {
                    return "PART %s:lastValue %s must be greater than %s".formatted(partName, fuzzyVariableDistributionPart.findLastValue(), previousFuzzyVariableDistributionPart.findLastValue());
                }
            }
            index++;
            previousFuzzyVariableDistributionPart = fuzzyVariableDistributionPart;
        }
        return null;
    }

    private String validateConcreteWeights(ConcreteWeights concreteWeights) {
        if (concreteWeights == null) {
            return "concreteWeights was null";
        }
        String error = this.validateConcreteWeight(concreteWeights.getChoice1());
        if (error != null) {
            return "choice1: " + error;
        }
        error = this.validateConcreteWeight(concreteWeights.getChoice2());
        if (error != null) {
            return "choice2: " + error;
        }
        error = this.validateConcreteWeight(concreteWeights.getChoice3());
        if (error != null) {
            return "choice3: " + error;
        }
        error = this.validateConcreteWeight(concreteWeights.getChoice4());
        if (error != null) {
            return "choice4: " + error;
        }
        if (concreteWeights.getChoice1() <= concreteWeights.getChoice2()) {
            return "choice1 must be greater than choice2";
        }
        if (concreteWeights.getChoice2() <= concreteWeights.getChoice3()) {
            return "choice2 must be greater than choice3";
        }
        if (concreteWeights.getChoice3() <= concreteWeights.getChoice4()) {
            return "choice3 must be greater than choice4";
        }
        double sum = concreteWeights.getChoice1() + concreteWeights.getChoice2() + concreteWeights.getChoice3() + concreteWeights.getChoice4();
        sum = (double) Math.round(sum * 100) / 100.0;
        if (sum != 1.0) {
            return "sum %s must be equal to 1.0".formatted(sum);
        }
        return null;
    }

    private String validateConcreteWeight(double concreteWeight) {
        if (concreteWeight < 0.1) {
            return "concreteWeight should be greater or equal than 0.1";
        }
        if (concreteWeight >= 1) {
            return "concreteWeight should be less than 1";
        }
        return null;
    }
}