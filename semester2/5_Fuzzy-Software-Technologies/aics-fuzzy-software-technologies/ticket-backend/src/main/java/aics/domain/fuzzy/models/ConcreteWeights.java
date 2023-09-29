package aics.domain.fuzzy.models;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain = true)
@AllArgsConstructor
@NoArgsConstructor
public class ConcreteWeights implements Serializable {
    private double choice1;
    private double choice2;
    private double choice3;
    private double choice4;
}
