package aics.domain.fuzzy;

import aics.domain.fuzzy.etities.FuzzyProfile;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import io.quarkus.panache.common.Parameters;
import io.quarkus.panache.common.Sort;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;
import java.util.Optional;

@ApplicationScoped
public class FuzzyProfileRepository implements PanacheRepository<FuzzyProfile> {
    public Optional<FuzzyProfile> findActive() {
        return find("active = true").singleResultOptional();
    }

    public Optional<FuzzyProfile> findByName(String name) {
        Parameters parameters = new Parameters();
        parameters.and("name", name);
        return find("name = :name", parameters).singleResultOptional();
    }


    public List<FuzzyProfile> findList() {
        return this.findAll(Sort.by("name", Sort.Direction.Ascending)).list();
    }

}