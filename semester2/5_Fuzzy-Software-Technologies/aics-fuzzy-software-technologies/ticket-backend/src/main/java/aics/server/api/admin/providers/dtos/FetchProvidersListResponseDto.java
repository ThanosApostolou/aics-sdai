package aics.server.api.admin.providers.dtos;

import aics.domain.provider.dtos.ProviderListItemDto;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.List;

@Data
@Accessors(chain = true)
public class FetchProvidersListResponseDto implements Serializable {
    private List<ProviderListItemDto> providers;
}
