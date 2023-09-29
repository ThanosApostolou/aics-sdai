import { Box, DialogContentText, FormControlLabel, FormGroup, Grid, Switch, TextField } from '@mui/material';
import { Fragment, useState } from 'react';
import { FuzzyProfileDto } from '../../../../modules/fuzzy/dtos/fuzzy-profile-dto';
import TopsisTable from '../../../../modules/fuzzy/components/TopsisTable';
import { FuzzySearchFiltersDto } from '../../../../modules/fuzzy/dtos/fuzzy-search-filters-dto';
import { RegularTopsisInfoDto } from '../../../../modules/fuzzy/dtos/regular-topsis-info-dto';

export interface TabRegularTopsisComponentProps {
    fuzzyProfileDto: FuzzyProfileDto;
    fuzzySearchFiltersDto: FuzzySearchFiltersDto,
    regularTopsisInfoDto: RegularTopsisInfoDto
}

export default function TabRegularTopsisComponent({ fuzzyProfileDto, fuzzySearchFiltersDto, regularTopsisInfoDto }: TabRegularTopsisComponentProps) {

    return (
        <Fragment>
            <Box style={{ width: '100%', height: '100%' }}>
                <h4>Step 1. Initial Data</h4>
                <TopsisTable dataTable={regularTopsisInfoDto.table1InitialData} yearCostCriteria={fuzzySearchFiltersDto.yearCostCriteria} durationCostCriteria={fuzzySearchFiltersDto.durationCostCriteria}></TopsisTable>

                <h4>Step 2. Normalized Data</h4>
                <TopsisTable dataTable={regularTopsisInfoDto.table2NormalizedData} yearCostCriteria={fuzzySearchFiltersDto.yearCostCriteria} durationCostCriteria={fuzzySearchFiltersDto.durationCostCriteria}></TopsisTable>

                <h4>Step 3. Weighted Normalized Data</h4>
                <TopsisTable dataTable={regularTopsisInfoDto.table3WeightedNormalizedData} yearCostCriteria={fuzzySearchFiltersDto.yearCostCriteria} durationCostCriteria={fuzzySearchFiltersDto.durationCostCriteria}></TopsisTable>

                <h4>Step 4. Final Data ordered by TOPSIS Score</h4>
                <TopsisTable dataTable={regularTopsisInfoDto.table4TopsisScoreData} yearCostCriteria={fuzzySearchFiltersDto.yearCostCriteria} durationCostCriteria={fuzzySearchFiltersDto.durationCostCriteria}></TopsisTable>
            </Box>
        </Fragment>
    );
}
