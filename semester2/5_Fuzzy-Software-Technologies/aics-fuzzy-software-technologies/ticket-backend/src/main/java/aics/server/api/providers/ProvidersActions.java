package aics.server.api.providers;

import aics.domain.provider.ProviderService;
import aics.domain.provider.dtos.ProviderDto;
import aics.domain.provider.entities.Provider;
import aics.infrastructure.errors.TicketException;
import aics.server.api.providers.dtos.FetchProvidersListResponseDto;
import io.quarkus.logging.Log;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.apache.commons.collections4.CollectionUtils;

import java.util.ArrayList;
import java.util.List;

@ApplicationScoped
public class ProvidersActions {
    @Inject
    ProviderService providerService;

    @Transactional(rollbackOn = Exception.class)
    public FetchProvidersListResponseDto doFetchProvidersList() throws TicketException {
        Log.info("Start ProvidersActions.doFetchProvidersList");
        FetchProvidersListResponseDto fetchProvidersListResponseDto = new FetchProvidersListResponseDto();
        List<Provider> providers = this.providerService.fetchAllProviders();
        List<ProviderDto> providersDtos = CollectionUtils.isNotEmpty(providers)
            ? providers.stream().map(ProviderDto::fromProvider).toList()
            : new ArrayList<>();

        fetchProvidersListResponseDto.setProviders(providersDtos);
        Log.info("End ProvidersActions.doFetchProvidersList");
        return fetchProvidersListResponseDto;
    }
}