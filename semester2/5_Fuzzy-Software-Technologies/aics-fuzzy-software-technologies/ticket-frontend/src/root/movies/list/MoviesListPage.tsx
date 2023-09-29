import React, { useEffect, useState } from 'react';
import { Typography, Divider, CircularProgress, Accordion, AccordionDetails, AccordionSummary, AccordionActions, Button } from '@mui/material';
import ScrollToTopOnMount from '../../shared/components/ScrollToTopOnMount';
import MoviesGridLayoutComponent from '../../../modules/movie/components/MoviesGridLayoutComponent';
import MovieIcon from '@mui/icons-material/Movie';
import { MovieListItemDto } from '../../../modules/movie/dtos/movie-list-item-dto';
import { useSnackbar } from 'notistack';
import { MoviesListService } from './movies-list-service';
import FuzzySearchAccordionComponent from './components/FuzzySearchAccordionComponent';
import { FuzzySearchFiltersDto } from '../../../modules/fuzzy/dtos/fuzzy-search-filters-dto';
import { FuzzySearchTopsisAnalysisDto } from '../../../modules/fuzzy/dtos/fuzzy-search-topsis-analysis-dto';

export default function MoviesListPage() {
    const [isWaitingFetch, setIsWaitingFetch] = useState<boolean>(false);
    const [movies, setMovies] = useState<MovieListItemDto[]>([]);
    const [fuzzySearchDebugInfoDto, setFuzzySearchDebugInfoDto] = useState<FuzzySearchTopsisAnalysisDto | null>(null);
    const [fuzzySearch, setFuzzySearch] = useState<boolean>(false);
    const { enqueueSnackbar } = useSnackbar();


    useEffect(() => {
        loadData(null);
    }, [])

    async function loadData(fuzzySearchFiltersDto: FuzzySearchFiltersDto | null) {
        setIsWaitingFetch(true);
        setMovies([]);
        setFuzzySearchDebugInfoDto(null);
        try {
            const fetchMoviesListResponseDto = await MoviesListService.fetchMoviesPlayingNow(fuzzySearchFiltersDto);
            console.log('fetchMoviesListResponseDto', fetchMoviesListResponseDto)
            setMovies(fetchMoviesListResponseDto.movies);
            setFuzzySearch(fetchMoviesListResponseDto.fuzzySearch);
            setFuzzySearchDebugInfoDto(fetchMoviesListResponseDto.fuzzySearchTopsisAnalysisDto);
            setIsWaitingFetch(false);
        } catch (e: any) {
            if (e?.response?.status === 422) {
                console.error(e?.response?.data?.error);
                enqueueSnackbar('Αποτυχημένη εύρεση λίστας ταινιών: ' + e?.response?.data?.error, { variant: 'error' })
            } else {
                console.error(e);
                enqueueSnackbar('Αποτυχημένη εύρεση λίστας ταινιών', { variant: 'error' });
            }
            setIsWaitingFetch(false);
        }

    }

    function handleFuzzySearch(fuzzySearchFiltersDto: FuzzySearchFiltersDto | null) {
        loadData(fuzzySearchFiltersDto);
    }

    return (
        <React.Fragment>
            <ScrollToTopOnMount />
            <div style={{
                display: 'flex',
                alignItems: 'center',
                flexWrap: 'wrap', marginTop: 10
            }}>
                <MovieIcon sx={{ marginLeft: 4 }} fontSize='large' />
                <Typography sx={{ fontSize: 'xx-large', marginLeft: 3, fontWeight: 'bolder' }}>ΠΑΙΖΟΝΤΑΙ ΤΩΡΑ</Typography>
            </div>
            <Divider variant="middle" style={{ marginBottom: 10 }} />

            <FuzzySearchAccordionComponent onSearch={handleFuzzySearch} fuzzySearchTopsisAnalysisDto={fuzzySearchDebugInfoDto}></FuzzySearchAccordionComponent>

            {isWaitingFetch
                ? (
                    <CircularProgress />
                )
                : (
                    <React.Fragment>

                        <MoviesGridLayoutComponent movies={movies} fuzzySearch={fuzzySearch} />
                    </React.Fragment>
                )}
        </React.Fragment>
    );
}
