package aics.domain.hall;

import aics.domain.hall.entities.Hall;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class HallRepository implements PanacheRepository<Hall> {

}