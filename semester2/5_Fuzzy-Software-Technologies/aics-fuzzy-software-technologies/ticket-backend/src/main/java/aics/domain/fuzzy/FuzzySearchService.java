package aics.domain.fuzzy;

import aics.domain.fuzzy.constants.FuzzySearchChoices;
import aics.domain.fuzzy.constants.FuzzyWeightsFields;
import aics.domain.fuzzy.dtos.*;
import aics.domain.fuzzy.etities.FuzzyProfile;
import aics.domain.fuzzy.models.*;
import aics.domain.movie.dtos.MovieListItemDto;
import aics.domain.movie.entities.Movie;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;

import java.util.*;
import java.util.stream.Collectors;

@ApplicationScoped
public class FuzzySearchService {
    @Inject
    FuzzyProfileService fuzzyProfileService;
    @Inject
    FuzzySearchValidator fuzzySearchValidator;

    public FuzzySearchResult fuzzySearch(List<Movie> movies, FuzzySearchFiltersDto fuzzySearchFiltersDto) {
        FuzzyProfile activeProfile = this.fuzzyProfileService.findActiveProfileOrDefault();

        Map<Long, Movie> idToMovieMap = new HashMap<>();
        for (Movie movie : movies) {
            idToMovieMap.put(movie.getMovieId(), movie);
        }

        // step1
        List<TopsisDataRow> table1InitialData = this.calculateTable1InitialDate(movies);

        if (activeProfile.isUseFuzzyTopsis()) {
            // FUZZY TOPSIS
            FuzzyWeights fuzzyWeights = activeProfile.getFuzzyProfileData().getFuzzyWeights();
            ConcreteWeights concreteWeights = activeProfile.getFuzzyProfileData().getConcreteWeights();
            EnumMap<FuzzySearchChoices, FuzzyWeightData> choiceToFuzzyWeightMap = this.getChoiceToFuzzyWeightMap(fuzzySearchFiltersDto, activeProfile.getFuzzyProfileData().getFuzzyWeights());
            // step2
            List<TopsisDataFuzzyRow> table2FuzzifiedData = this.calculateTable2FuzzifiedData(table1InitialData, activeProfile.getFuzzyProfileData());
            // step3
            List<TopsisDataFuzzyRow> table3FuzzifiedDistributionData = this.calculateTable3FuzzifiedDistributionData(table2FuzzifiedData, activeProfile.getFuzzyProfileData());
            // step4
            List<TopsisDataFuzzyRow> table4NormalizedData = this.calculateTable4NormalizedData(table3FuzzifiedDistributionData, fuzzySearchFiltersDto);
            // step5
            List<TopsisDataFuzzyRow> table5WeightedDistributionData = this.calculateTable5WeightedDistributionData(table4NormalizedData, choiceToFuzzyWeightMap);
            // step6
            Triple<List<TopsisDataFuzzyRow>, TopsisDataFuzzyRow, TopsisDataFuzzyRow> table6TopsisScoreResult = this.calculateTable6TopsisScore(table5WeightedDistributionData);
            // generate FUZZY TOPSIS analysis
            FuzzySearchTopsisAnalysisDto createFuzzySearchTopsisAnalysisDto = this.createFuzzySearchTopsisAnalysisDtoForFuzzyTopsis(activeProfile, fuzzySearchFiltersDto, choiceToFuzzyWeightMap, table1InitialData, table2FuzzifiedData, table3FuzzifiedDistributionData, table4NormalizedData, table5WeightedDistributionData, table6TopsisScoreResult);


            List<MovieListItemDto> movieDtos = new ArrayList<>();
            for (TopsisDataFuzzyRow topsisDataFuzzyRow : table6TopsisScoreResult.getLeft()) {
                Movie movie = idToMovieMap.get(topsisDataFuzzyRow.getMovieId());
                movieDtos.add(MovieListItemDto.fromMovieAndTopsisScore(movie, topsisDataFuzzyRow.getScore()));
            }

            return new FuzzySearchResult(
                    movieDtos,
                    createFuzzySearchTopsisAnalysisDto
            );
        } else {
            // REGULAR TOPSIS
            ConcreteWeights concreteWeights = activeProfile.getFuzzyProfileData().getConcreteWeights();
            EnumMap<FuzzySearchChoices, Double> choiceToWeightMap = this.getChoiceToWeightMap(fuzzySearchFiltersDto, concreteWeights);
            // step2
            List<TopsisDataRow> table2NormalizedData = this.calculateTable2NormalizedData(table1InitialData);
            // step3
            List<TopsisDataRow> table3WeightedNormalizedData = this.calculateTable3WeightedNormalizedData(table2NormalizedData, choiceToWeightMap);
            // step4
            Triple<List<TopsisDataRow>, TopsisDataRow, TopsisDataRow> calculateTable4TopsisScoreResult = this.calculateTable4TopsisScore(table3WeightedNormalizedData, fuzzySearchFiltersDto.isYearCostCriteria(), fuzzySearchFiltersDto.isDurationCostCriteria());
            List<TopsisDataRow> table4TopsisScore = calculateTable4TopsisScoreResult.getLeft();
            TopsisDataRow bestRow = calculateTable4TopsisScoreResult.getMiddle();
            TopsisDataRow worstRow = calculateTable4TopsisScoreResult.getRight();
            // generate TOPSIS analysis
            FuzzySearchTopsisAnalysisDto createFuzzySearchTopsisAnalysisDto = this.createFuzzySearchTopsisAnalysisDtoForRegularTopsis(activeProfile, fuzzySearchFiltersDto, choiceToWeightMap, table1InitialData, table2NormalizedData, table3WeightedNormalizedData, table4TopsisScore, bestRow, worstRow);

            List<MovieListItemDto> movieDtos = new ArrayList<>();
            for (TopsisDataRow topsisDataRow : table4TopsisScore) {
                Movie movie = idToMovieMap.get(topsisDataRow.getMovieId());
                movieDtos.add(MovieListItemDto.fromMovieAndTopsisScore(movie, topsisDataRow.getScore()));
            }

            return new FuzzySearchResult(
                    movieDtos,
                    createFuzzySearchTopsisAnalysisDto
            );
        }
    }

    private List<TopsisDataRow> calculateTable1InitialDate(List<Movie> movies) {
        return movies.stream().map(movie -> new TopsisDataRow(
                movie.getMovieId(),
                movie.getName(),
                movie.getRating(),
                movie.getPopularity(),
                movie.getYear(),
                movie.getDuration(),
                0, 0, 0)).collect(Collectors.toList()
        );
    }

    private List<TopsisDataRow> calculateTable2NormalizedData(final List<TopsisDataRow> table1InitialData) {
        List<TopsisDataRow> table2NormalizedData = new ArrayList<>();

        double allRatingSquaredSum = 0;
        double allPopularitySquaredSum = 0;
        double allYearSquaredSum = 0;
        double allDurationSquaredSum = 0;
        for (int i = 0; i < table1InitialData.size(); i++) {
            allRatingSquaredSum += Math.pow(table1InitialData.get(i).getRating(), 2);
            allPopularitySquaredSum += Math.pow(table1InitialData.get(i).getPopularity(), 2);
            allYearSquaredSum += Math.pow(table1InitialData.get(i).getYear(), 2);
            allDurationSquaredSum += Math.pow(table1InitialData.get(i).getDuration(), 2);
        }

        for (int i = 0; i < table1InitialData.size(); i++) {
            final TopsisDataRow currentDataRow = table1InitialData.get(i);

            final double rating = currentDataRow.getRating();
            final double popularity = currentDataRow.getPopularity();
            final double year = currentDataRow.getYear();
            final double duration = currentDataRow.getDuration();
            TopsisDataRow normalizedDataRow = new TopsisDataRow(
                    currentDataRow.getMovieId(),
                    currentDataRow.getName(),
                    rating / Math.sqrt(allRatingSquaredSum),
                    popularity / Math.sqrt(allPopularitySquaredSum),
                    year / Math.sqrt(allYearSquaredSum),
                    duration / Math.sqrt(allDurationSquaredSum),
                    0, 0, 0
            );
            table2NormalizedData.add(normalizedDataRow);
        }
        return table2NormalizedData;
    }

    private List<TopsisDataRow> calculateTable3WeightedNormalizedData(final List<TopsisDataRow> table2NormalizedData,
                                                                      final EnumMap<FuzzySearchChoices, Double> choiceToWeightMap) {
        List<TopsisDataRow> table3WeightedNormalizedData = new ArrayList<>();

        for (int i = 0; i < table2NormalizedData.size(); i++) {
            final TopsisDataRow currentDataRow = table2NormalizedData.get(i);
            final double newRating = currentDataRow.getRating() * choiceToWeightMap.get(FuzzySearchChoices.RATING);
            final double newPopularity = currentDataRow.getPopularity() * choiceToWeightMap.get(FuzzySearchChoices.POPULARITY);
            final double newYear = currentDataRow.getYear() * choiceToWeightMap.get(FuzzySearchChoices.YEAR);
            final double newDuration = currentDataRow.getDuration() * choiceToWeightMap.get(FuzzySearchChoices.DURATION);
            TopsisDataRow weightedNormalizedDataRow = new TopsisDataRow(
                    currentDataRow.getMovieId(),
                    currentDataRow.getName(),
                    newRating,
                    newPopularity,
                    newYear,
                    newDuration,
                    0, 0, 0
            );
            table3WeightedNormalizedData.add(weightedNormalizedDataRow);
        }

        return table3WeightedNormalizedData;
    }


    private Triple<List<TopsisDataRow>, TopsisDataRow, TopsisDataRow> calculateTable4TopsisScore(final List<TopsisDataRow> table3WeightedNormalizedData, boolean yearCostCriteria, boolean durationCostCriteria) {
        List<TopsisDataRow> table4TopsisScore = new ArrayList<>();

        double ratingBest = table3WeightedNormalizedData.stream().map(TopsisDataRow::getRating).max(Double::compare).get();
        double ratingWorst = table3WeightedNormalizedData.stream().map(TopsisDataRow::getRating).min(Double::compare).get();
        double popularityBest = table3WeightedNormalizedData.stream().map(TopsisDataRow::getPopularity).min(Double::compare).get();
        double popularityWorst = table3WeightedNormalizedData.stream().map(TopsisDataRow::getPopularity).max(Double::compare).get();
        double yearBest = yearCostCriteria
                ? table3WeightedNormalizedData.stream().map(TopsisDataRow::getYear).min(Double::compare).get()
                : table3WeightedNormalizedData.stream().map(TopsisDataRow::getYear).max(Double::compare).get();
        double yearWorst = yearCostCriteria
                ? table3WeightedNormalizedData.stream().map(TopsisDataRow::getYear).max(Double::compare).get()
                : table3WeightedNormalizedData.stream().map(TopsisDataRow::getYear).min(Double::compare).get();
        double durationBest = durationCostCriteria
                ? table3WeightedNormalizedData.stream().map(TopsisDataRow::getDuration).min(Double::compare).get()
                : table3WeightedNormalizedData.stream().map(TopsisDataRow::getDuration).max(Double::compare).get();
        double durationWorst = durationCostCriteria
                ? table3WeightedNormalizedData.stream().map(TopsisDataRow::getDuration).max(Double::compare).get()
                : table3WeightedNormalizedData.stream().map(TopsisDataRow::getDuration).min(Double::compare).get();

        for (int i = 0; i < table3WeightedNormalizedData.size(); i++) {
            final TopsisDataRow currentDataRow = table3WeightedNormalizedData.get(i);
            final double dpos = Math.sqrt(
                    Math.pow(currentDataRow.getRating() - ratingBest, 2)
                            + Math.pow(currentDataRow.getPopularity() - popularityBest, 2)
                            + Math.pow(currentDataRow.getYear() - yearBest, 2)
                            + Math.pow(currentDataRow.getDuration() - durationBest, 2)
            );
            final double dneg = Math.sqrt(
                    Math.pow(currentDataRow.getRating() - ratingWorst, 2)
                            + Math.pow(currentDataRow.getPopularity() - popularityWorst, 2)
                            + Math.pow(currentDataRow.getYear() - yearWorst, 2)
                            + Math.pow(currentDataRow.getDuration() - durationWorst, 2)
            );
            final double score = dneg / (dpos + dneg);
            TopsisDataRow normalizedDataRow = new TopsisDataRow(
                    currentDataRow.getMovieId(),
                    currentDataRow.getName(),
                    currentDataRow.getRating(),
                    currentDataRow.getPopularity(),
                    currentDataRow.getYear(),
                    currentDataRow.getDuration(),
                    dpos, dneg, score
            );
            table4TopsisScore.add(normalizedDataRow);
        }
        table4TopsisScore.sort(Comparator.comparing(TopsisDataRow::getScore).reversed());
        TopsisDataRow bestRow = new TopsisDataRow(
                -1L,
                "BEST",
                ratingBest,
                popularityBest,
                yearBest,
                durationBest,
                0, 0, 0
        );
        TopsisDataRow worstRow = new TopsisDataRow(
                -2L,
                "WORST",
                ratingWorst,
                popularityWorst,
                yearWorst,
                durationWorst,
                0, 0, 0
        );
        return Triple.of(table4TopsisScore, bestRow, worstRow);
    }

    private FuzzySearchTopsisAnalysisDto createFuzzySearchTopsisAnalysisDtoForRegularTopsis(FuzzyProfile activeProfile,
                                                                                            FuzzySearchFiltersDto fuzzySearchFiltersDto,
                                                                                            EnumMap<FuzzySearchChoices, Double> choiceToWeightMap,
                                                                                            List<TopsisDataRow> table1InitialData,
                                                                                            List<TopsisDataRow> table2NormalizedData, List<TopsisDataRow> table3WeightedNormalizedData, List<TopsisDataRow> table4TopsisScore, TopsisDataRow bestRow, TopsisDataRow worstRow) {
        final double roundFactor = 1000.0;
        String weightRating = String.valueOf(Math.round(choiceToWeightMap.get(FuzzySearchChoices.RATING) * roundFactor) / roundFactor);
        String weightPopularity = String.valueOf(Math.round(choiceToWeightMap.get(FuzzySearchChoices.POPULARITY) * roundFactor) / roundFactor);
        String weightYear = String.valueOf(Math.round(choiceToWeightMap.get(FuzzySearchChoices.YEAR) * roundFactor) / roundFactor);
        String weightDuration = String.valueOf(Math.round(choiceToWeightMap.get(FuzzySearchChoices.DURATION) * roundFactor) / roundFactor);

        TopsisDataRowDto weightsRow = new TopsisDataRowDto(0L, "WEIGHTS", weightRating, weightPopularity, weightYear, weightDuration, "", "", "");
        // table1
        List<TopsisDataRowDto> table1InitialDataDtos = table1InitialData.stream().map(topsisDataRow -> TopsisDataRowDto.fromTopsisDataRow(topsisDataRow, false, false, false)).collect(Collectors.toList());
        table1InitialDataDtos.add(weightsRow);
        TopsisDataTableDto table1InitialDataDto = new TopsisDataTableDto(table1InitialDataDtos, false, false, false);

        // table2
        List<TopsisDataRowDto> table2NormalizedDataDtos = table2NormalizedData.stream().map(topsisDataRow -> TopsisDataRowDto.fromTopsisDataRow(topsisDataRow, false, false, false)).collect(Collectors.toList());
        table2NormalizedDataDtos.add(weightsRow);
        TopsisDataTableDto table2NormalizedDataDto = new TopsisDataTableDto(table2NormalizedDataDtos, false, false, false);
        // table3
        List<TopsisDataRowDto> table3WeightedNormalizedDataDtos = table3WeightedNormalizedData.stream().map(topsisDataRow -> TopsisDataRowDto.fromTopsisDataRow(topsisDataRow, false, false, false)).collect(Collectors.toList());
        TopsisDataTableDto table3WeightedNormalizedDataDto = new TopsisDataTableDto(table3WeightedNormalizedDataDtos, false, false, false);
        // table4
        List<TopsisDataRowDto> table4TopsisScoreDtos = table4TopsisScore.stream().map(topsisDataRow -> TopsisDataRowDto.fromTopsisDataRow(topsisDataRow, true, true, true)).collect(Collectors.toList());
        table4TopsisScoreDtos.add(TopsisDataRowDto.fromTopsisDataRow(bestRow, false, false, false));
        table4TopsisScoreDtos.add(TopsisDataRowDto.fromTopsisDataRow(worstRow, false, false, false));
        TopsisDataTableDto table4TopsisScoreDataDto = new TopsisDataTableDto(table4TopsisScoreDtos, true, true, true);

        RegularTopsisInfoDto regularTopsisInfoDto = new RegularTopsisInfoDto(table1InitialDataDto, table2NormalizedDataDto, table3WeightedNormalizedDataDto, table4TopsisScoreDataDto);

        return new FuzzySearchTopsisAnalysisDto(FuzzyProfileDto.fromFuzzyProfile(activeProfile),
                fuzzySearchFiltersDto,
                regularTopsisInfoDto,
                null);
    }

    private FuzzySearchTopsisAnalysisDto createFuzzySearchTopsisAnalysisDtoForFuzzyTopsis(FuzzyProfile activeProfile,
                                                                                          FuzzySearchFiltersDto fuzzySearchFiltersDto,
                                                                                          EnumMap<FuzzySearchChoices, FuzzyWeightData> choiceToFuzzyWeightMap,
                                                                                          List<TopsisDataRow> table1InitialData,
                                                                                          List<TopsisDataFuzzyRow> table2FuzzifiedData,
                                                                                          List<TopsisDataFuzzyRow> table3FuzzifiedDistributionData,
                                                                                          List<TopsisDataFuzzyRow> table4NormalizedData,
                                                                                          List<TopsisDataFuzzyRow> table5WeightedDistributionData,
                                                                                          Triple<List<TopsisDataFuzzyRow>, TopsisDataFuzzyRow, TopsisDataFuzzyRow> table6TopsisScoreResult) {
        final double roundFactor = 1000.0;
        FuzzyWeightData weightRating = choiceToFuzzyWeightMap.get(FuzzySearchChoices.RATING);
        FuzzyWeightData weightPopularity = choiceToFuzzyWeightMap.get(FuzzySearchChoices.POPULARITY);
        FuzzyWeightData weightYear = choiceToFuzzyWeightMap.get(FuzzySearchChoices.YEAR);
        FuzzyWeightData weightDuration = choiceToFuzzyWeightMap.get(FuzzySearchChoices.DURATION);

        TopsisDataRowDto weightsRow = new TopsisDataRowDto(0L, "FUZZY WEIGHTS", weightRating.name(), weightPopularity.name(), weightYear.name(), weightDuration.name(), "", "", "");
        // table1
        List<TopsisDataRowDto> table1InitialDataDtos = table1InitialData.stream().map(topsisDataRow -> TopsisDataRowDto.fromTopsisDataRow(topsisDataRow, false, false, false)).collect(Collectors.toList());
        table1InitialDataDtos.add(weightsRow);
        TopsisDataTableDto table1InitialDataDto = new TopsisDataTableDto(table1InitialDataDtos, false, false, false);

        // table2
        List<TopsisDataRowDto> table2FuzzifiedDataDtos = table2FuzzifiedData.stream().map(topsisDataFuzzyRow -> TopsisDataRowDto.fromTopsisDataFuzzyRow(topsisDataFuzzyRow, false, false, false)).collect(Collectors.toList());
        table2FuzzifiedDataDtos.add(new TopsisDataRowDto(0L, "FUZZY WEIGHTS", weightRating.fuzzyValueMx().toFormattedString(), weightPopularity.fuzzyValueMx().toFormattedString(), weightYear.fuzzyValueMx().toFormattedString(), weightDuration.fuzzyValueMx().toFormattedString(), "", "", ""));
        TopsisDataTableDto table2FuzzifiedDataDto = new TopsisDataTableDto(table2FuzzifiedDataDtos, false, false, false);
        // table3
        List<TopsisDataRowDto> table3FuzzifiedDistributionDataDtos = table3FuzzifiedDistributionData.stream().map(topsisDataFuzzyRow -> TopsisDataRowDto.fromTopsisDataFuzzyRow(topsisDataFuzzyRow, false, false, false)).collect(Collectors.toList());
        table3FuzzifiedDistributionDataDtos.add(new TopsisDataRowDto(0L, "FUZZY WEIGHTS", weightRating.fuzzyValue().toFormattedString(), weightPopularity.fuzzyValue().toFormattedString(), weightYear.fuzzyValue().toFormattedString(), weightDuration.fuzzyValue().toFormattedString(), "", "", ""));
        TopsisDataTableDto table3FuzzifiedDistributionDataDto = new TopsisDataTableDto(table3FuzzifiedDistributionDataDtos, false, false, false);
        // table4
        List<TopsisDataRowDto> table4NormalizedDataDtos = table4NormalizedData.stream().map(topsisDataFuzzyRow -> TopsisDataRowDto.fromTopsisDataFuzzyRow(topsisDataFuzzyRow, false, false, false)).collect(Collectors.toList());
        table4NormalizedDataDtos.add(new TopsisDataRowDto(0L, "FUZZY WEIGHTS", weightRating.fuzzyValue().toFormattedString(), weightPopularity.fuzzyValue().toFormattedString(), weightYear.fuzzyValue().toFormattedString(), weightDuration.fuzzyValue().toFormattedString(), "", "", ""));
        TopsisDataTableDto table4NormalizedDataDto = new TopsisDataTableDto(table4NormalizedDataDtos, false, false, false);
        // table5
        List<TopsisDataRowDto> table5WeightedDistributionDataDtos = table5WeightedDistributionData.stream().map(topsisDataFuzzyRow -> TopsisDataRowDto.fromTopsisDataFuzzyRow(topsisDataFuzzyRow, false, false, false)).collect(Collectors.toList());
        TopsisDataTableDto table5WeightedDistributionDataDto = new TopsisDataTableDto(table5WeightedDistributionDataDtos, false, false, false);
        // table6
        List<TopsisDataFuzzyRow> table6TopsisScore = table6TopsisScoreResult.getLeft();
        TopsisDataFuzzyRow fpisRow = table6TopsisScoreResult.getMiddle();
        TopsisDataFuzzyRow fnisRow = table6TopsisScoreResult.getRight();
        List<TopsisDataRowDto> table6TopsisScoreDtos = table6TopsisScore.stream().map(topsisDataFuzzyRow -> TopsisDataRowDto.fromTopsisDataFuzzyRow(topsisDataFuzzyRow, true, true, true)).collect(Collectors.toList());
        // FPIS row
        table6TopsisScoreDtos.add(new TopsisDataRowDto(fpisRow.getMovieId(), fpisRow.getName(), fpisRow.getRating().toFormattedString(), fpisRow.getPopularity().toFormattedString(), fpisRow.getYear().toFormattedString(), fpisRow.getDuration().toFormattedString(), "", "", ""));
        // FNIS row
        table6TopsisScoreDtos.add(new TopsisDataRowDto(fnisRow.getMovieId(), fnisRow.getName(), fnisRow.getRating().toFormattedString(), fnisRow.getPopularity().toFormattedString(), fnisRow.getYear().toFormattedString(), fnisRow.getDuration().toFormattedString(), "", "", ""));
        TopsisDataTableDto table6TopsisScoreDto = new TopsisDataTableDto(table6TopsisScoreDtos, true, true, true);

        FuzzyTopsisInfoDto fuzzyTopsisInfoDto = new FuzzyTopsisInfoDto(table1InitialDataDto, table2FuzzifiedDataDto, table3FuzzifiedDistributionDataDto, table4NormalizedDataDto, table5WeightedDistributionDataDto, table6TopsisScoreDto);

        return new FuzzySearchTopsisAnalysisDto(FuzzyProfileDto.fromFuzzyProfile(activeProfile),
                fuzzySearchFiltersDto,
                null,
                fuzzyTopsisInfoDto);
    }


    private List<TopsisDataFuzzyRow> calculateTable2FuzzifiedData(List<TopsisDataRow> table1InitialData,
                                                                  FuzzyProfileData fuzzyProfileData
    ) {
        List<TopsisDataFuzzyRow> table2FuzzifiedData = new ArrayList<>();
        for (TopsisDataRow topsisDataRow : table1InitialData) {
            FuzzyValue ratingValue = FuzzyVariableI.fuzzyValueMx(fuzzyProfileData.getFuzzyVariableRating(), topsisDataRow.getRating());
            FuzzyValue popularityValue = FuzzyVariableI.fuzzyValueMx(fuzzyProfileData.getFuzzyVariablePopularity(), topsisDataRow.getPopularity());
            FuzzyValue yearValue = FuzzyVariableI.fuzzyValueMx(fuzzyProfileData.getFuzzyVariableYear(), topsisDataRow.getYear());
            FuzzyValue durationValue = FuzzyVariableI.fuzzyValueMx(fuzzyProfileData.getFuzzyVariableDuration(), topsisDataRow.getDuration());
            TopsisDataFuzzyRow topsisDataFuzzyRow = new TopsisDataFuzzyRow(
                    topsisDataRow.getMovieId(),
                    topsisDataRow.getName(),
                    ratingValue,
                    popularityValue,
                    yearValue,
                    durationValue,
                    0, 0, 0);
            table2FuzzifiedData.add(topsisDataFuzzyRow);
        }
        return table2FuzzifiedData;
    }

    private List<TopsisDataFuzzyRow> calculateTable3FuzzifiedDistributionData(List<TopsisDataFuzzyRow> table2FuzzifiedData,
                                                                              FuzzyProfileData fuzzyProfileData
    ) {
        List<TopsisDataFuzzyRow> table3WeightedFuzzifiedData = new ArrayList<>();
        for (TopsisDataFuzzyRow topsisDataFuzzyRow : table2FuzzifiedData) {
            TopsisDataFuzzyRow newTopsisDataFuzzyRow = new TopsisDataFuzzyRow(
                    topsisDataFuzzyRow.getMovieId(),
                    topsisDataFuzzyRow.getName(),
                    FuzzyVariableI.fuzzyValueFromVariableAndValueMx(fuzzyProfileData.getFuzzyVariableRating(), topsisDataFuzzyRow.getRating()),
                    FuzzyVariableI.fuzzyValueFromVariableAndValueMx(fuzzyProfileData.getFuzzyVariablePopularity(), topsisDataFuzzyRow.getPopularity()),
                    FuzzyVariableI.fuzzyValueFromVariableAndValueMx(fuzzyProfileData.getFuzzyVariableYear(), topsisDataFuzzyRow.getYear()),
                    FuzzyVariableI.fuzzyValueFromVariableAndValueMx(fuzzyProfileData.getFuzzyVariableDuration(), topsisDataFuzzyRow.getDuration()),
                    0, 0, 0);
            table3WeightedFuzzifiedData.add(newTopsisDataFuzzyRow);
        }
        return table3WeightedFuzzifiedData;
    }

    private List<TopsisDataFuzzyRow> calculateTable4NormalizedData(List<TopsisDataFuzzyRow> table3FuzzifiedDistributionData,
                                                                   FuzzySearchFiltersDto fuzzySearchFiltersDto
    ) {
        double ratingFactor = table3FuzzifiedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getRating().max()).max(Double::compareTo).get();
        double popularityFactor = table3FuzzifiedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getRating().min()).min(Double::compareTo).get();
        double yearFactor = fuzzySearchFiltersDto.isYearCostCriteria()
                ? table3FuzzifiedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getYear().min()).min(Double::compareTo).get()
                : table3FuzzifiedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getYear().max()).max(Double::compareTo).get();
        double durationFactor = fuzzySearchFiltersDto.isDurationCostCriteria()
                ? table3FuzzifiedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getDuration().min()).min(Double::compareTo).get()
                : table3FuzzifiedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getDuration().max()).max(Double::compareTo).get();

        List<TopsisDataFuzzyRow> table4NormalizedData = new ArrayList<>();
        for (TopsisDataFuzzyRow topsisDataFuzzyRow : table3FuzzifiedDistributionData) {
            FuzzyValue newRating = new FuzzyValue(topsisDataFuzzyRow.getRating().a() / ratingFactor,
                    topsisDataFuzzyRow.getRating().b() / ratingFactor,
                    topsisDataFuzzyRow.getRating().c() / ratingFactor,
                    topsisDataFuzzyRow.getRating().d() / ratingFactor);

            FuzzyValue newPopularity = new FuzzyValue(popularityFactor / topsisDataFuzzyRow.getPopularity().d(),
                    popularityFactor / topsisDataFuzzyRow.getPopularity().c(),
                    popularityFactor / topsisDataFuzzyRow.getPopularity().b(),
                    popularityFactor / topsisDataFuzzyRow.getPopularity().a());

            FuzzyValue newYear = fuzzySearchFiltersDto.isYearCostCriteria()
                    ? new FuzzyValue(yearFactor / topsisDataFuzzyRow.getYear().d(),
                    yearFactor / topsisDataFuzzyRow.getYear().c(),
                    yearFactor / topsisDataFuzzyRow.getYear().b(),
                    yearFactor / topsisDataFuzzyRow.getYear().a())
                    : new FuzzyValue(topsisDataFuzzyRow.getYear().a() / yearFactor,
                    topsisDataFuzzyRow.getYear().b() / yearFactor,
                    topsisDataFuzzyRow.getYear().c() / yearFactor,
                    topsisDataFuzzyRow.getYear().d() / yearFactor);

            FuzzyValue newDuration = fuzzySearchFiltersDto.isDurationCostCriteria()
                    ? new FuzzyValue(durationFactor / topsisDataFuzzyRow.getDuration().d(),
                    durationFactor / topsisDataFuzzyRow.getDuration().c(),
                    durationFactor / topsisDataFuzzyRow.getDuration().b(),
                    durationFactor / topsisDataFuzzyRow.getDuration().a())
                    : new FuzzyValue(topsisDataFuzzyRow.getDuration().a() / durationFactor,
                    topsisDataFuzzyRow.getDuration().b() / durationFactor,
                    topsisDataFuzzyRow.getDuration().c() / durationFactor,
                    topsisDataFuzzyRow.getDuration().d() / durationFactor);

            TopsisDataFuzzyRow newTopsisDataFuzzyRow = new TopsisDataFuzzyRow(
                    topsisDataFuzzyRow.getMovieId(),
                    topsisDataFuzzyRow.getName(),
                    newRating,
                    newPopularity,
                    newYear,
                    newDuration,
                    0, 0, 0);
            table4NormalizedData.add(newTopsisDataFuzzyRow);
        }
        return table4NormalizedData;
    }

    private List<TopsisDataFuzzyRow> calculateTable5WeightedDistributionData(List<TopsisDataFuzzyRow> table2FuzzifiedData,
                                                                             EnumMap<FuzzySearchChoices, FuzzyWeightData> choiceToFuzzyWeightMap
    ) {
        FuzzyValue ratingWeight = choiceToFuzzyWeightMap.get(FuzzySearchChoices.RATING).fuzzyValue();
        FuzzyValue popularityWeight = choiceToFuzzyWeightMap.get(FuzzySearchChoices.POPULARITY).fuzzyValue();
        FuzzyValue yearWeight = choiceToFuzzyWeightMap.get(FuzzySearchChoices.YEAR).fuzzyValue();
        FuzzyValue durationWeight = choiceToFuzzyWeightMap.get(FuzzySearchChoices.DURATION).fuzzyValue();
        List<TopsisDataFuzzyRow> table5WeightedDistributionData = new ArrayList<>();
        for (TopsisDataFuzzyRow topsisDataFuzzyRow1 : table2FuzzifiedData) {
            TopsisDataFuzzyRow topsisDataFuzzyRow = new TopsisDataFuzzyRow(
                    topsisDataFuzzyRow1.getMovieId(),
                    topsisDataFuzzyRow1.getName(),
                    FuzzyValue.multiply(topsisDataFuzzyRow1.getRating(), ratingWeight),
                    FuzzyValue.multiply(topsisDataFuzzyRow1.getPopularity(), popularityWeight),
                    FuzzyValue.multiply(topsisDataFuzzyRow1.getYear(), yearWeight),
                    FuzzyValue.multiply(topsisDataFuzzyRow1.getDuration(), durationWeight),
                    0, 0, 0);
            table5WeightedDistributionData.add(topsisDataFuzzyRow);
        }
        return table5WeightedDistributionData;
    }


    private Triple<List<TopsisDataFuzzyRow>, TopsisDataFuzzyRow, TopsisDataFuzzyRow> calculateTable6TopsisScore(List<TopsisDataFuzzyRow> table5WeightedDistributionData) {
        double ratingMax = table5WeightedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getRating().max()).max(Double::compareTo).get();
        double ratingMin = table5WeightedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getRating().min()).min(Double::compareTo).get();
        double popularityMax = table5WeightedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getPopularity().max()).max(Double::compareTo).get();
        double popularityMin = table5WeightedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getPopularity().min()).min(Double::compareTo).get();
        double yearMax = table5WeightedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getYear().max()).max(Double::compareTo).get();
        double yearMin = table5WeightedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getYear().min()).min(Double::compareTo).get();
        double durationMax = table5WeightedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getDuration().max()).max(Double::compareTo).get();
        double durationMin = table5WeightedDistributionData.stream().map(topsisDataFuzzyRow -> topsisDataFuzzyRow.getDuration().min()).min(Double::compareTo).get();

        FuzzyValue ratingFpis = new FuzzyValue(ratingMax, ratingMax, ratingMax, ratingMax);
        FuzzyValue ratingFnis = new FuzzyValue(ratingMin, ratingMin, ratingMin, ratingMin);
        FuzzyValue popularityFpis = new FuzzyValue(popularityMax, popularityMax, popularityMax, popularityMax);
        FuzzyValue popularityFnis = new FuzzyValue(popularityMin, popularityMin, popularityMin, popularityMin);
        FuzzyValue yearFpis = new FuzzyValue(yearMax, yearMax, yearMax, yearMax);
        FuzzyValue yearFnis = new FuzzyValue(yearMin, yearMin, yearMin, yearMin);
        FuzzyValue durationFpis = new FuzzyValue(durationMax, durationMax, durationMax, durationMax);
        FuzzyValue durationFnis = new FuzzyValue(durationMin, durationMin, durationMin, durationMin);

        List<TopsisDataFuzzyRow> table6TopsisScore = new ArrayList<>();
        for (TopsisDataFuzzyRow topsisDataFuzzyRow : table5WeightedDistributionData) {
            double dpos = FuzzyValue.calculateDistance(topsisDataFuzzyRow.getRating(), ratingFpis)
                    + FuzzyValue.calculateDistance(topsisDataFuzzyRow.getPopularity(), popularityFpis)
                    + FuzzyValue.calculateDistance(topsisDataFuzzyRow.getYear(), yearFpis)
                    + FuzzyValue.calculateDistance(topsisDataFuzzyRow.getDuration(), durationFpis);
            double dneg = FuzzyValue.calculateDistance(topsisDataFuzzyRow.getRating(), ratingFnis)
                    + FuzzyValue.calculateDistance(topsisDataFuzzyRow.getPopularity(), popularityFnis)
                    + FuzzyValue.calculateDistance(topsisDataFuzzyRow.getYear(), yearFnis)
                    + FuzzyValue.calculateDistance(topsisDataFuzzyRow.getDuration(), durationFnis);
            double score = dneg / (dneg + dpos);
            TopsisDataFuzzyRow newTopsisDataFuzzyRow = new TopsisDataFuzzyRow(
                    topsisDataFuzzyRow.getMovieId(),
                    topsisDataFuzzyRow.getName(),
                    topsisDataFuzzyRow.getRating(),
                    topsisDataFuzzyRow.getPopularity(),
                    topsisDataFuzzyRow.getYear(),
                    topsisDataFuzzyRow.getDuration(),
                    dpos, dneg, score);
            table6TopsisScore.add(newTopsisDataFuzzyRow);
        }
        table6TopsisScore.sort(Comparator.comparing(TopsisDataFuzzyRow::getScore).reversed());

        TopsisDataFuzzyRow fpisRow = new TopsisDataFuzzyRow(-1L, "FPIS",
                ratingFpis, popularityFpis, yearFpis, durationFpis,
                0, 0, 0);
        TopsisDataFuzzyRow fnisRow = new TopsisDataFuzzyRow(-2L, "FNIS",
                ratingFnis, popularityFnis, yearFnis, durationFnis,
                0, 0, 0);
        return ImmutableTriple.of(table6TopsisScore, fpisRow, fnisRow);
    }

    private EnumMap<FuzzySearchChoices, Double> getChoiceToWeightMap(FuzzySearchFiltersDto fuzzySearchFiltersDto, ConcreteWeights concreteWeights) {
        EnumMap<FuzzySearchChoices, Double> choiceToWeightMap = new EnumMap<>(FuzzySearchChoices.class);
        choiceToWeightMap.put(fuzzySearchFiltersDto.getChoice1(), concreteWeights.getChoice1());
        choiceToWeightMap.put(fuzzySearchFiltersDto.getChoice2(), concreteWeights.getChoice2());
        choiceToWeightMap.put(fuzzySearchFiltersDto.getChoice3(), concreteWeights.getChoice3());
        choiceToWeightMap.put(fuzzySearchFiltersDto.getChoice4(), concreteWeights.getChoice4());
        return choiceToWeightMap;

    }


    private EnumMap<FuzzySearchChoices, FuzzyWeightData> getChoiceToFuzzyWeightMap(FuzzySearchFiltersDto fuzzySearchFiltersDto, FuzzyWeights fuzzyWeights) {
        EnumMap<FuzzySearchChoices, FuzzyWeightData> choiceToWeightMap = new EnumMap<>(FuzzySearchChoices.class);

        choiceToWeightMap.put(
                fuzzySearchFiltersDto.getChoice1(),
                new FuzzyWeightData(
                        FuzzyWeightsFields.VERY_HIGH_IMPORTANCE.name(),
                        new FuzzyValue(0, 0, 0, 1),
                        FuzzyVariableI.fuzzyValueFromVariableAndValueMx(fuzzyWeights, new FuzzyValue(0, 0, 0, 1))
                )
        );
        choiceToWeightMap.put(
                fuzzySearchFiltersDto.getChoice2(),
                new FuzzyWeightData(
                        FuzzyWeightsFields.HIGH_IMPORTANCE.name(),
                        new FuzzyValue(0, 0, 1, 0),
                        FuzzyVariableI.fuzzyValueFromVariableAndValueMx(fuzzyWeights, new FuzzyValue(0, 0, 1, 0))
                )
        );
        choiceToWeightMap.put(
                fuzzySearchFiltersDto.getChoice3(),
                new FuzzyWeightData(FuzzyWeightsFields.AVERAGE_IMPORTANCE.name(),
                        new FuzzyValue(0, 1, 0, 0),
                        FuzzyVariableI.fuzzyValueFromVariableAndValueMx(fuzzyWeights, new FuzzyValue(0, 1, 0, 0))
                )
        );
        choiceToWeightMap.put(
                fuzzySearchFiltersDto.getChoice4(),
                new FuzzyWeightData(FuzzyWeightsFields.LOW_IMPORTANCE.name(),
                        new FuzzyValue(1, 0, 0, 0),
                        FuzzyVariableI.fuzzyValueFromVariableAndValueMx(fuzzyWeights, new FuzzyValue(1, 0, 0, 0))
                )
        );
        return choiceToWeightMap;

    }

    private static record FuzzyWeightData(String name, FuzzyValue fuzzyValueMx, FuzzyValue fuzzyValue) {

    }
}