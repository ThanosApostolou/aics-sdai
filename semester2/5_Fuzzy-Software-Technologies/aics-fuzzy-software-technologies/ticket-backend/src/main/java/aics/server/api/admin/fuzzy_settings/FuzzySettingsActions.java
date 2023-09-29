package aics.server.api.admin.fuzzy_settings;

import aics.domain.fuzzy.FuzzyProfileRepository;
import aics.domain.fuzzy.FuzzyProfileService;
import aics.domain.fuzzy.dtos.FuzzyProfileDto;
import aics.domain.fuzzy.etities.FuzzyProfile;
import aics.infrastructure.errors.TicketErrorStatus;
import aics.infrastructure.errors.TicketException;
import aics.server.api.admin.fuzzy_settings.dtos.CreateFuzzyProfileResponseDto;
import aics.server.api.admin.fuzzy_settings.dtos.DeleteFuzzyProfileResponseDto;
import aics.server.api.admin.fuzzy_settings.dtos.FetchFuzzyProfilesResponseDto;
import aics.server.api.admin.fuzzy_settings.dtos.UpdateFuzzyProfileResponseDto;
import io.quarkus.logging.Log;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;

@ApplicationScoped
public class FuzzySettingsActions {
    @Inject
    FuzzyProfileRepository fuzzyProfileRepository;
    @Inject
    FuzzyProfileService fuzzyProfileService;

    @Transactional(rollbackOn = Exception.class)
    public FetchFuzzyProfilesResponseDto doFetchFuzzyProfiles() throws TicketException {
        Log.info("Start FuzzySettingsActions.doFetchFuzzyProfileNames");

        List<FuzzyProfile> foundFuzzyProfiles = fuzzyProfileRepository.findList();
        boolean foundActive = foundFuzzyProfiles.stream().anyMatch(FuzzyProfile::isActive);

        FuzzyProfile defaultProfile = this.fuzzyProfileService.createDefaultProfile(!foundActive);

        List<FuzzyProfile> allFuzzyProfiles = new ArrayList<>(List.of(defaultProfile));
        allFuzzyProfiles.addAll(foundFuzzyProfiles);

        List<String> fuzzyProfilesNames = allFuzzyProfiles.stream().map(FuzzyProfile::getName).collect(Collectors.toList());
        LinkedHashMap<String, FuzzyProfileDto> fuzzyProfilesMap = new LinkedHashMap<>();
        for (FuzzyProfile fuzzyProfile : allFuzzyProfiles) {
            fuzzyProfilesMap.put(fuzzyProfile.getName(), FuzzyProfileDto.fromFuzzyProfile(fuzzyProfile));
        }

        FetchFuzzyProfilesResponseDto fetchProvidersListResponseDto = new FetchFuzzyProfilesResponseDto(
                fuzzyProfilesNames,
                fuzzyProfilesMap
        );
        Log.info("End FuzzySettingsActions.doFetchFuzzyProfileNames");
        return fetchProvidersListResponseDto;
    }


    @Transactional(rollbackOn = Exception.class)
    public CreateFuzzyProfileResponseDto doCreateFuzzyProfile(FuzzyProfileDto fuzzyProfileDto) throws TicketException {
        Log.info("Start FuzzySettingsActions.doCreateFuzzyProfile");

        String error = this.fuzzyProfileService.createFuzzyProfile(fuzzyProfileDto);
        if (error != null) {
            throw new TicketException(new Exception(error), error, TicketErrorStatus.UNPROCESSABLE_ENTITY_422);
        }

        Log.info("End FuzzySettingsActions.doCreateFuzzyProfile");
        return new CreateFuzzyProfileResponseDto(fuzzyProfileDto.getName(), null);
    }

    @Transactional(rollbackOn = Exception.class)
    public UpdateFuzzyProfileResponseDto doUpdateFuzzyProfile(FuzzyProfileDto fuzzyProfileDto) throws TicketException {
        Log.info("Start FuzzySettingsActions.doUpdateFuzzyProfile");

        String error = this.fuzzyProfileService.updateFuzzyProfile(fuzzyProfileDto);
        if (error != null) {
            throw new TicketException(new Exception(error), error, TicketErrorStatus.UNPROCESSABLE_ENTITY_422);
        }

        Log.info("End FuzzySettingsActions.doUpdateFuzzyProfile");
        return new UpdateFuzzyProfileResponseDto(fuzzyProfileDto.getName(), null);
    }

    @Transactional(rollbackOn = Exception.class)
    public DeleteFuzzyProfileResponseDto doDeleteFuzzyProfile(String name) throws TicketException {
        Log.info("Start FuzzySettingsActions.doDeleteFuzzyProfile");

        String error = this.fuzzyProfileService.deleteFuzzyProfileByName(name);
        if (error != null) {
            throw new TicketException(new Exception(error), error, TicketErrorStatus.UNPROCESSABLE_ENTITY_422);
        }

        Log.info("End FuzzySettingsActions.doDeleteFuzzyProfile");
        return new DeleteFuzzyProfileResponseDto(null);
    }
}