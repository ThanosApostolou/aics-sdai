package aics.domain.fuzzy.dtos;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.List;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
@AllArgsConstructor
public class TopsisDataTableDto implements Serializable {
    private List<TopsisDataRowDto> rows;
    private boolean showDpos;
    private boolean showDneg;
    private boolean showScore;
}
