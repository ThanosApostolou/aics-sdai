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
public class FuzzyTopsisInfoDto implements Serializable {
    private TopsisDataTableDto table1InitialData;
    private TopsisDataTableDto table2FuzzifiedData;
    private TopsisDataTableDto table3FuzzifiedDistributionDataDto;
    private TopsisDataTableDto table4NormalizedDataDto;
    private TopsisDataTableDto table5WeightedDistributionDataDto;
    private TopsisDataTableDto table6TopsisScoreDto;
}
