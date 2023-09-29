import { Box, DialogContentText, FormControlLabel, FormGroup, Grid, Switch, TextField } from '@mui/material';
import { Fragment, useState } from 'react';
import ConcreteWeightsComponent from '../../../admin/fuzzy-settings/components/ConcreteWeightsComponent';
import FuzzyVarComponent from '../../../../modules/fuzzy/components/FuzzyVarComponent';
import { FuzzyProfileDto } from '../../../../modules/fuzzy/dtos/fuzzy-profile-dto';
import { ConcreteWeights } from '../../../../modules/fuzzy/models/concrete-weights';
import { FuzzyVariableDuration } from '../../../../modules/fuzzy/models/fuzzy-variable-duration';
import { FuzzyVariablePopularity } from '../../../../modules/fuzzy/models/fuzzy-variable-popularity';
import { FuzzyVariableRating } from '../../../../modules/fuzzy/models/fuzzy-variable-rating';
import { FuzzyVariableYear } from '../../../../modules/fuzzy/models/fuzzy-variable-year';
import { FuzzyWeights } from '../../../../modules/fuzzy/models/fuzzy-weights';
import ConfirmationDialogComponent from '../../../../modules/ui/components/MovieDialogDeleteComponent';

export interface TabFuzzyProfileComponentProps {
    fuzzyProfileDto: FuzzyProfileDto;
}

export default function TabFuzzyProfileComponent({ fuzzyProfileDto }: TabFuzzyProfileComponentProps) {
    // const [fuzzyProfileDto, setFuzzyProfileDto] = useState<FuzzyProfileDto>(fuzzyProfileDto)
    const [fuzzyVariableYear, setFuzzyVariableYear] = useState<FuzzyVariableYear>(fuzzyProfileDto.fuzzyProfileData.fuzzyVariableYear);
    const [fuzzyVariableRating, setFuzzyVariableRating] = useState<FuzzyVariableRating>(fuzzyProfileDto.fuzzyProfileData.fuzzyVariableRating);
    const [fuzzyVariablePopularity, setFuzzyVariablePopularity] = useState<FuzzyVariablePopularity>(fuzzyProfileDto.fuzzyProfileData.fuzzyVariablePopularity);
    const [fuzzyVariableDuration, setFuzzyVariableDuration] = useState<FuzzyVariableDuration>(fuzzyProfileDto.fuzzyProfileData.fuzzyVariableDuration);
    const [fuzzyWeights, setFuzzyWeights] = useState<FuzzyWeights>(fuzzyProfileDto.fuzzyProfileData.fuzzyWeights);
    const [concreteWeights, setConcreteWeights] = useState<ConcreteWeights>(fuzzyProfileDto.fuzzyProfileData.concreteWeights);
    const [name, setName] = useState<string>(fuzzyProfileDto.name)
    const [showTopsisAnalysis, setShowTopsisAnalysis] = useState<boolean>(fuzzyProfileDto.showTopsisAnalysis)
    const [active, setActive] = useState<boolean>(fuzzyProfileDto.active);
    const [useFuzzyTopsis, setUseFuzzyTopsis] = useState<boolean>(fuzzyProfileDto.useFuzzyTopsis);
    const [deleteProfileConfirmationDialogOpen, setDeleteProfileConfirmationDialogOpen] = useState(false);

    return (
        <Fragment>
            <Box style={{ width: '100%', height: '100%' }}>
                <Grid container direction="row" padding={2}
                    justifyContent="space-between"
                    alignItems="center">
                    <Grid item>
                        <h2>Fuzzy Profile: {fuzzyProfileDto.name}</h2>
                        {fuzzyProfileDto.name === '' && (
                            <TextField size='small' label="Όνομα" value={name} onChange={(e) => setName(e.target.value)} />
                        )}
                    </Grid>

                </Grid>
                <FormGroup>
                    <FormControlLabel disabled={true} control={<Switch />} label="ShowTopsisAnalysis" checked={showTopsisAnalysis} />
                    <FormControlLabel disabled={true} control={<Switch />} label="Active" checked={active} />
                    <FormControlLabel disabled={true} control={<Switch />} label="UseFuzzyTopsis" checked={useFuzzyTopsis} />
                </FormGroup>
                <hr></hr>

                <FuzzyVarComponent fuzzyVariable={fuzzyVariableYear} readonly={true} xStepSize={5}></FuzzyVarComponent>
                <hr></hr>
                <br></br>

                <FuzzyVarComponent fuzzyVariable={fuzzyVariableRating} readonly={true} xStepSize={1}></FuzzyVarComponent>
                <hr></hr>
                <br></br>

                <FuzzyVarComponent fuzzyVariable={fuzzyVariablePopularity} readonly={true} xStepSize={10}></FuzzyVarComponent>
                <hr></hr>
                <br></br>

                <FuzzyVarComponent fuzzyVariable={fuzzyVariableDuration} readonly={true} xStepSize={20}></FuzzyVarComponent>
                <hr></hr>
                <br></br>

                <FuzzyVarComponent fuzzyVariable={fuzzyWeights} readonly={true} xStepSize={1} numberStep={0.1}></FuzzyVarComponent>
                <ConcreteWeightsComponent concreteWeights={concreteWeights} readonly={true}></ConcreteWeightsComponent>

            </Box>
            <ConfirmationDialogComponent open={deleteProfileConfirmationDialogOpen} onCancel={() => setDeleteProfileConfirmationDialogOpen(false)}>
                <DialogContentText>
                    Είστε σίγουρος ότι θέλετε να διαγράψεται αυτό το Fuzzy Profile?
                </DialogContentText>
            </ConfirmationDialogComponent>
        </Fragment>
    );
}
