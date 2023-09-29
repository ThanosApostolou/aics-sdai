package aics.domain.fuzzy.etities;

import aics.domain.fuzzy.models.FuzzyProfileData;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

@Entity(name = "FUZZY_PROFILE")
@Getter
@Setter
@Accessors(chain = true)
public class FuzzyProfile {
    @Id()
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "FUZZY_PROFILE_ID")
    private Long fuzzyProfileId;
    @Column(name = "NAME", nullable = false, unique = true)
    private String name;
    @Column(name = "FUZZY_PROFILE_DATA", columnDefinition = "json", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private FuzzyProfileData fuzzyProfileData;
    @Column(name = "SHOW_TOPSIS_ANALYSIS", nullable = false)
    private boolean showTopsisAnalysis;
    @Column(name = "ACTIVE", nullable = false)
    private boolean active;
    @Column(name = "USE_FUZZY_TOPSIS", nullable = false)
    private boolean useFuzzyTopsis;

}
