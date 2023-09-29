import { Box, Grid } from '@mui/material';
import { Fragment, useEffect, useState } from 'react';
import { ChartData } from 'chart.js';
import { FuzzyService } from '../fuzzy-service';
import GraphFuzzyDistributionComponent from './GraphFuzzyDistributionComponent';
import { FuzzyVariablePartPosition } from '../fuzzy-constants';
import { FuzzyVariableDistributionPart, FuzzyVariableI } from '../models/fuzzy-variable-distribution';
import FuzzyVarPartComponent from './FuzzyVarPartComponent';

export interface FuzzyVarComponentProps {
    fuzzyVariable: FuzzyVariableI;
    readonly: boolean;
    xStepSize: number;
    numberStep?: number;
}

export default function FuzzyVarComponent({ fuzzyVariable, readonly, xStepSize, numberStep }: FuzzyVarComponentProps) {
    const [fuzzyVariableChartData, setFuzzyVariableChartData] = useState<ChartData<"line", { x: number, y: number }[], number> | null>(FuzzyService.convertFuzzyVariableToChartData(fuzzyVariable));
    const [fuzzyVartPart1, setFuzzyVartPart1] = useState<FuzzyVariableDistributionPart>(fuzzyVariable.get1stPart());
    const [fuzzyVartPart2, setFuzzyVartPart2] = useState<FuzzyVariableDistributionPart>(fuzzyVariable.get2ndPart());
    const [fuzzyVartPart3, setFuzzyVartPart3] = useState<FuzzyVariableDistributionPart>(fuzzyVariable.get3rdPart());
    const [fuzzyVartPart4, setFuzzyVartPart4] = useState<FuzzyVariableDistributionPart>(fuzzyVariable.get4thPart());
    const [name, setName] = useState<string>(fuzzyVariable.getName());

    function fuzzyVarPart1Updated(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        setFuzzyVartPart1(fuzzyVariableDistributionPart);
        fuzzyVariable.set1stPart(fuzzyVariableDistributionPart);
        setFuzzyVariableChartData(FuzzyService.convertFuzzyVariableToChartData(fuzzyVariable));
    }

    function fuzzyVarPart2Updated(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        setFuzzyVartPart2(fuzzyVariableDistributionPart);
        fuzzyVariable.set2ndPart(fuzzyVariableDistributionPart);
        setFuzzyVariableChartData(FuzzyService.convertFuzzyVariableToChartData(fuzzyVariable));
    }

    function fuzzyVarPart3Updated(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        setFuzzyVartPart3(fuzzyVariableDistributionPart);
        fuzzyVariable.set3rdPart(fuzzyVariableDistributionPart);
        setFuzzyVariableChartData(FuzzyService.convertFuzzyVariableToChartData(fuzzyVariable));
    }

    function fuzzyVarPart4Updated(fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) {
        setFuzzyVartPart4(fuzzyVariableDistributionPart);
        fuzzyVariable.set4thPart(fuzzyVariableDistributionPart);
        setFuzzyVariableChartData(FuzzyService.convertFuzzyVariableToChartData(fuzzyVariable));
    }


    return (
        <Fragment>
            <Box style={{ width: '100%', height: '100%' }}>
                <h3 style={{ textAlign: 'center' }}>{name} Variable</h3>
                <Grid container spacing={2} sx={{ padding: 1 }}>
                    <Grid item container xs={12} lg={6} boxShadow={1}>
                        <FuzzyVarPartComponent fuzzyVariableDistributionPart={fuzzyVartPart1} readonly={readonly} fuzzyVariablePartPosition={FuzzyVariablePartPosition.START}
                            numberStep={numberStep}
                            fuzzyVarPartUpdated={fuzzyVarPart1Updated}></FuzzyVarPartComponent>
                    </Grid>
                    <Grid item container xs={12} lg={6} boxShadow={1}>
                        <FuzzyVarPartComponent fuzzyVariableDistributionPart={fuzzyVartPart2} readonly={readonly} fuzzyVariablePartPosition={FuzzyVariablePartPosition.MIDDLE}
                            numberStep={numberStep}
                            fuzzyVarPartUpdated={fuzzyVarPart2Updated}></FuzzyVarPartComponent>
                    </Grid>
                    <Grid item container xs={12} lg={6} boxShadow={1}>
                        <FuzzyVarPartComponent fuzzyVariableDistributionPart={fuzzyVartPart3} readonly={readonly} fuzzyVariablePartPosition={FuzzyVariablePartPosition.MIDDLE}
                            numberStep={numberStep}
                            fuzzyVarPartUpdated={fuzzyVarPart3Updated}></FuzzyVarPartComponent>
                    </Grid>
                    <Grid item container xs={12} lg={6} boxShadow={1}>
                        <FuzzyVarPartComponent fuzzyVariableDistributionPart={fuzzyVartPart4} readonly={readonly} fuzzyVariablePartPosition={FuzzyVariablePartPosition.END}
                            numberStep={numberStep}
                            fuzzyVarPartUpdated={fuzzyVarPart4Updated}></FuzzyVarPartComponent>
                    </Grid>
                </Grid>

                {fuzzyVariableChartData && (
                    <div>
                        <GraphFuzzyDistributionComponent datasetIdKey={name + 'Chart'} xTitle={name} chartData={fuzzyVariableChartData} xStepSize={xStepSize}></GraphFuzzyDistributionComponent>
                    </div>
                )}
            </Box>
        </Fragment>
    );
}
