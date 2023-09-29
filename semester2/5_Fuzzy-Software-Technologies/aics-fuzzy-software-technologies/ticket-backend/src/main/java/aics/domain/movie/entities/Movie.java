package aics.domain.movie.entities;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.type.SqlTypes;

import java.time.LocalDateTime;

@Entity(name = "MOVIES")
@Getter
@Setter
@Accessors(chain = true)
public class Movie {
    @Id()
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "MOVIE_ID")
    private Long movieId;
    @Column(name = "NAME", nullable = false, length = 255)
    private String name;
    @Column(name = "DESCRIPTION", nullable = true, length = 2048)
    private String description;
    @Column(name = "IMAGE", nullable = false, columnDefinition = "longblob")
    @JdbcTypeCode(SqlTypes.BLOB)
    private byte[] image;
    @Column(name = "IMAGE_NAME", nullable = false)
    private String imageName;
    @Column(name = "IMAGE_MIME_PREFIX", nullable = false)
    private String imageMimePrefix;
    @Column(name = "DIRECTORS", nullable = false, length = 255)
    private String directors;
    @Column(name = "SCRIPT", nullable = false, length = 255)
    private String script;
    @Column(name = "ACTORS", nullable = false, length = 255)
    private String actors;
    @Column(name = "APPROPRIATENESS", nullable = false, length = 255)
    private String appropriateness;
    @Column(name = "TRAILER_SRC_URL", nullable = false, length = 255)
    private String trailerSrcUrl;
    @Column(name = "DURATION", nullable = false, length = 255)
    private int duration;
    @Column(name = "YEAR", nullable = false)
    private int year;
    @Column(name = "RATING", nullable = false)
    private double rating;
    @Column(name = "POPULARITY", nullable = false)
    private int popularity;
    @Column(name = "CREATED_ON", nullable = false)
    @CreationTimestamp
    private LocalDateTime createdOn;
    @Column(name = "UPDATED_ON", nullable = false)
    @UpdateTimestamp
    private LocalDateTime updatedOn;
}
