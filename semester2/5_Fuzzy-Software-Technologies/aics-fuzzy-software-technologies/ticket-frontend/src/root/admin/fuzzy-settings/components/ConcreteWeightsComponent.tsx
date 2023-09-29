import { Box, Grid, TextField } from '@mui/material';
import { ChangeEvent, Fragment, useEffect, useState } from 'react';
import { useSnackbar } from 'notistack';
import { ConcreteWeights } from '../../../../modules/fuzzy/models/concrete-weights';

export interface ConcreteWeightsComponentProps {
    concreteWeights: ConcreteWeights;
    readonly: boolean;
    concreteWeightsUpdated?: (concreteWeights: ConcreteWeights) => void;
}

export default function ConcreteWeightsComponent({ concreteWeights, readonly, concreteWeightsUpdated }: ConcreteWeightsComponentProps) {
    const [choice1, setChoice1] = useState<number>(concreteWeights.choice1);
    const [choice2, setChoice2] = useState<number>(concreteWeights.choice2);
    const [choice3, setChoice3] = useState<number>(concreteWeights.choice3);
    const [choice4, setChoice4] = useState<number>(concreteWeights.choice4);


    const { enqueueSnackbar } = useSnackbar();

    useEffect(() => {
        // setFuzzyVariableYearChartData(FuzzyService.convertFuzzyVariableToChartData(fuzzyProfileDto.fuzzyProfileData.fuzzyVariableYear));
        // setFuzzyVariableRatingChartData(FuzzyService.convertFuzzyVariableToChartData(fuzzyProfileDto.fuzzyProfileData.fuzzyVariableRating));
        // setFuzzyVariablePopularityChartData(FuzzyService.convertFuzzyVariableToChartData(fuzzyProfileDto.fuzzyProfileData.fuzzyVariablePopularity));
    }, [])

    function choice1Updated(e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
        const newChoice1 = e.target.value ? parseFloat(e.target.value) : 0;
        setChoice1(newChoice1);
        concreteWeights.choice1 = newChoice1;
        if (concreteWeightsUpdated) {
            concreteWeightsUpdated(concreteWeights)
        }
    }

    function choice2Updated(e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
        const newChoice2 = e.target.value ? parseFloat(e.target.value) : 0;
        setChoice2(newChoice2);
        concreteWeights.choice2 = newChoice2;
        if (concreteWeightsUpdated) {
            concreteWeightsUpdated(concreteWeights)
        }
    }

    function choice3Updated(e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
        const newChoice3 = e.target.value ? parseFloat(e.target.value) : 0;
        setChoice3(newChoice3);
        concreteWeights.choice3 = newChoice3;
        if (concreteWeightsUpdated) {
            concreteWeightsUpdated(concreteWeights)
        }
    }

    function choice4Updated(e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
        const newChoice4 = e.target.value ? parseFloat(e.target.value) : 0;
        setChoice4(newChoice4);
        concreteWeights.choice4 = newChoice4;
        if (concreteWeightsUpdated) {
            concreteWeightsUpdated(concreteWeights)
        }
    }

    return (
        <Fragment>
            <Box style={{ width: '100%', height: '100%' }}>
                <h3 style={{ textAlign: 'center' }}>Concrete Weights</h3>
                <Grid container spacing={2} sx={{ padding: 1 }}>
                    <Grid item container xs={6} lg={3}>
                        <TextField disabled={readonly} type='number' inputProps={{ step: 0.01, min: 0, max: 1 }} label="Choice1" value={choice1} onChange={choice1Updated} />
                    </Grid>
                    <Grid item container xs={6} lg={3}>
                        <TextField disabled={readonly} type='number' inputProps={{ step: 0.01, min: 0, max: 1 }} label="Choice2" value={choice2} onChange={choice2Updated} />
                    </Grid>
                    <Grid item container xs={6} lg={3}>
                        <TextField disabled={readonly} type='number' inputProps={{ step: 0.01, min: 0, max: 1 }} label="Choice3" value={choice3} onChange={choice3Updated} />
                    </Grid>
                    <Grid item container xs={6} lg={3}>
                        <TextField disabled={readonly} type='number' inputProps={{ step: 0.01, min: 0, max: 1 }} label="Choice4" value={choice4} onChange={choice4Updated} />
                    </Grid>
                </Grid>
            </Box>
        </Fragment>
    );
}
