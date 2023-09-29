package aics.domain.fuzzy.dtos;

import aics.domain.fuzzy.etities.FuzzyProfile;
import aics.domain.fuzzy.models.FuzzyProfileData;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
public class FuzzyProfileDto implements Serializable {
    private Long fuzzyProfileId;
    private String name;
    private FuzzyProfileData fuzzyProfileData;
    private boolean showTopsisAnalysis;
    private boolean active;
    private boolean useFuzzyTopsis;

    public static FuzzyProfileDto fromFuzzyProfile(FuzzyProfile fuzzyProfile) {
        if (fuzzyProfile == null) {
            return null;
        }
        return new FuzzyProfileDto()
                .setFuzzyProfileId(fuzzyProfile.getFuzzyProfileId())
                .setName(fuzzyProfile.getName())
                .setFuzzyProfileData(fuzzyProfile.getFuzzyProfileData())
                .setShowTopsisAnalysis(fuzzyProfile.isShowTopsisAnalysis())
                .setActive(fuzzyProfile.isActive())
                .setUseFuzzyTopsis(fuzzyProfile.isUseFuzzyTopsis());
    }
}
