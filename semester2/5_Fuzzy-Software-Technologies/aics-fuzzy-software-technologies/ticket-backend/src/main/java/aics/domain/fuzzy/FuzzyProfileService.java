package aics.domain.fuzzy;

import aics.domain.fuzzy.constants.*;
import aics.domain.fuzzy.dtos.FuzzyProfileDto;
import aics.domain.fuzzy.etities.FuzzyProfile;
import aics.domain.fuzzy.models.*;
import io.smallrye.common.constraint.NotNull;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.apache.commons.lang3.StringUtils;

@ApplicationScoped
public class FuzzyProfileService {
    @Inject
    FuzzyProfileRepository fuzzyProfileRepository;
    @Inject
    FuzzyProfileValidator fuzzyProfileValidator;

    public FuzzyProfile createDefaultProfile(boolean isDefaultActive) {
        FuzzyVariableYear fuzzyVariableYear = new FuzzyVariableYear(
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariableYearFields.OLD.name(), null, 1950, 1990, 2010.0),
                new FuzzyVariableDistributionPartTriangular(FuzzyVariableYearFields.RECENT.name(), 1990.0, 2010, 2018.0),
                new FuzzyVariableDistributionPartTriangular(FuzzyVariableYearFields.NEW.name(), 2010.0, 2018, 2024.0),
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariableYearFields.VERY_NEW.name(), 2018.0, 2024, 2030, null)
        );

        FuzzyVariableRating fuzzyVariableRating = new FuzzyVariableRating(
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariableRatingFields.BAD.name(), null, 1, 3, 5.0),
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariableRatingFields.AVERAGE.name(), 4.0, 5, 6, 7.0),
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariableRatingFields.GOOD.name(), 6.0, 7, 8, 9.0),
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariableRatingFields.VERY_GOOD.name(), 8.0, 9, 10, null)
        );

        FuzzyVariablePopularity fuzzyVariablePopularity = new FuzzyVariablePopularity(
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariablePopularityFields.VERY_POPULAR.name(), null, 1, 50, 100.0),
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariablePopularityFields.POPULAR.name(), 50.0, 100, 160, 250.0),
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariablePopularityFields.AVERAGE.name(), 160.0, 250, 350, 500.0),
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariablePopularityFields.UNPOPULAR.name(), 350.0, 500, 800, null)
        );

        FuzzyVariableDuration fuzzyVariableDuration = new FuzzyVariableDuration(
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariableDurationFields.SMALL.name(), null, 1, 20, 60.0),
                new FuzzyVariableDistributionPartTriangular(FuzzyVariableDurationFields.AVERAGE.name(), 20.0, 60, 120.0),
                new FuzzyVariableDistributionPartTriangular(FuzzyVariableDurationFields.BIG.name(), 60.0, 120, 160.0),
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyVariableDurationFields.HUGE.name(), 120.0, 160, 300, null)
        );

        FuzzyWeights fuzzyWeights = new FuzzyWeights(
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyWeightsFields.LOW_IMPORTANCE.name(), null, 0.1, 0.3, 0.5),
                new FuzzyVariableDistributionPartTriangular(FuzzyWeightsFields.AVERAGE_IMPORTANCE.name(), 0.3, 0.5, 0.7),
                new FuzzyVariableDistributionPartTriangular(FuzzyWeightsFields.HIGH_IMPORTANCE.name(), 0.5, 0.7, 0.9),
                new FuzzyVariableDistributionPartTrapezoidal(FuzzyWeightsFields.VERY_HIGH_IMPORTANCE.name(), 0.7, 0.9, 1.0, null)
        );

        ConcreteWeights concreteWeights = new ConcreteWeights(0.4, 0.3, 0.2, 0.1);

        FuzzyProfileData fuzzyProfileData = new FuzzyProfileData()
                .setFuzzyVariableYear(fuzzyVariableYear)
                .setFuzzyVariableRating(fuzzyVariableRating)
                .setFuzzyVariablePopularity(fuzzyVariablePopularity)
                .setFuzzyVariableDuration(fuzzyVariableDuration)
                .setFuzzyWeights(fuzzyWeights)
                .setConcreteWeights(concreteWeights);

        return new FuzzyProfile()
                .setName(FuzzyConstants.DEFAULT)
                .setShowTopsisAnalysis(true)
                .setActive(isDefaultActive)
                .setUseFuzzyTopsis(false)
                .setFuzzyProfileData(fuzzyProfileData);
    }

    public String createFuzzyProfile(FuzzyProfileDto fuzzyProfileDto) {
        String error = this.fuzzyProfileValidator.validateForCreateFuzzyProfile(fuzzyProfileDto);
        if (error != null) {
            return error;
        }

        FuzzyProfile existingFuzzyProfile = this.fuzzyProfileRepository.findByName(fuzzyProfileDto.getName()).orElse(null);
        if (existingFuzzyProfile != null) {
            return "Profile with name %s already exists".formatted(fuzzyProfileDto.getName());
        }
        this.resetActiveIfNewActivePorfile(fuzzyProfileDto.isActive(), fuzzyProfileDto.getName());

        FuzzyProfile newFuzzyProfile = new FuzzyProfile()
                .setName(fuzzyProfileDto.getName())
                .setShowTopsisAnalysis(fuzzyProfileDto.isShowTopsisAnalysis())
                .setActive(fuzzyProfileDto.isActive())
                .setUseFuzzyTopsis(fuzzyProfileDto.isUseFuzzyTopsis())
                .setFuzzyProfileData(fuzzyProfileDto.getFuzzyProfileData());

        this.fuzzyProfileRepository.persist(newFuzzyProfile);

        return null;

    }


    public String updateFuzzyProfile(FuzzyProfileDto fuzzyProfileDto) {
        String error = this.fuzzyProfileValidator.validateForUpdateFuzzyProfile(fuzzyProfileDto);
        if (error != null) {
            return error;
        }

        FuzzyProfile existingFuzzyProfile = this.fuzzyProfileRepository.findByName(fuzzyProfileDto.getName()).orElse(null);
        if (existingFuzzyProfile == null) {
            return "Could not find Profile with name %s".formatted(fuzzyProfileDto.getName());
        }
        if (!fuzzyProfileDto.getFuzzyProfileId().equals(existingFuzzyProfile.getFuzzyProfileId())) {
            return "fuzzyProfileId is different";
        }
        this.resetActiveIfNewActivePorfile(fuzzyProfileDto.isActive(), fuzzyProfileDto.getName());
        existingFuzzyProfile
                .setShowTopsisAnalysis(fuzzyProfileDto.isShowTopsisAnalysis())
                .setActive(fuzzyProfileDto.isActive())
                .setUseFuzzyTopsis(fuzzyProfileDto.isUseFuzzyTopsis())
                .setFuzzyProfileData(fuzzyProfileDto.getFuzzyProfileData());

        this.fuzzyProfileRepository.persist(existingFuzzyProfile);

        return null;

    }

    private void resetActiveIfNewActivePorfile(boolean active, String name) {
        if (active) {
            FuzzyProfile existingActiveProfile = this.fuzzyProfileRepository.findActive().orElse(null);
            if (existingActiveProfile != null && !StringUtils.equals(name, existingActiveProfile.getName())) {
                existingActiveProfile.setActive(false);
                this.fuzzyProfileRepository.persist(existingActiveProfile);
            }
        }
    }

    public String deleteFuzzyProfileByName(String name) {
        if (StringUtils.isEmpty(name)) {
            return "name cannot be empty";
        }
        if (StringUtils.equals(name, FuzzyConstants.DEFAULT) || StringUtils.equals(name, FuzzyConstants.NEW)) {
            return "name cannot DEFAULT or NEW";
        }

        FuzzyProfile existingFuzzyProfile = this.fuzzyProfileRepository.findByName(name).orElse(null);
        if (existingFuzzyProfile == null) {
            return "Cannot find profile with name %s".formatted(name);
        }

        this.fuzzyProfileRepository.delete(existingFuzzyProfile);

        return null;

    }

    public @NotNull FuzzyProfile findActiveProfileOrDefault() {
        return this.fuzzyProfileRepository.findActive().orElseGet(() -> this.createDefaultProfile(true));
    }
}