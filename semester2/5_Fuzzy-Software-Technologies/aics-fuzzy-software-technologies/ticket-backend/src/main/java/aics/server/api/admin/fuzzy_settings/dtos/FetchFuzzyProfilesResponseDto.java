package aics.server.api.admin.fuzzy_settings.dtos;

import aics.domain.fuzzy.dtos.FuzzyProfileDto;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.List;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
public class FetchFuzzyProfilesResponseDto implements Serializable {
    final List<String> fuzzyProfilesNames;
    final LinkedHashMap<String, FuzzyProfileDto> fuzzyProfilesMap;
}
