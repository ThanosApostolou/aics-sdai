package aics.server.api.admin.fuzzy_settings.dtos;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
public class DeleteFuzzyProfileResponseDto implements Serializable {
    final Serializable error;
}
