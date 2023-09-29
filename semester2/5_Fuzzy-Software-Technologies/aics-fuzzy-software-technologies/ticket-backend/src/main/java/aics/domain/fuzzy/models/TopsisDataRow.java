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
public class TopsisDataRow implements Serializable {
    private Long movieId;
    private String name;
    private double rating;
    private double popularity;
    private double year;
    private double duration;
    private double dpos;
    private double dneg;
    private double score;
}
