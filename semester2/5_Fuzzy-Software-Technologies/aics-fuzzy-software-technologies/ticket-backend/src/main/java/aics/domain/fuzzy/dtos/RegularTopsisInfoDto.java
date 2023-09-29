package aics.domain.fuzzy.dtos;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
@AllArgsConstructor
public class RegularTopsisInfoDto implements Serializable {
    private TopsisDataTableDto table1InitialData;
    private TopsisDataTableDto table2NormalizedData;
    private TopsisDataTableDto table3WeightedNormalizedData;
    private TopsisDataTableDto table4TopsisScoreData;
}
