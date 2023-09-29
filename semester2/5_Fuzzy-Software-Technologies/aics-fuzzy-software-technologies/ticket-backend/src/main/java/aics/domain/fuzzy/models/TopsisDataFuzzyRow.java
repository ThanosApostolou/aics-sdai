package aics.domain.fuzzy.models;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain = true)
@RequiredArgsConstructor
@AllArgsConstructor
public class TopsisDataFuzzyRow implements Serializable {
    private Long movieId;
    private String name;
    private FuzzyValue rating;
    private FuzzyValue popularity;
    private FuzzyValue year;
    private FuzzyValue duration;
    private double dpos;
    private double dneg;
    private double score;
}
