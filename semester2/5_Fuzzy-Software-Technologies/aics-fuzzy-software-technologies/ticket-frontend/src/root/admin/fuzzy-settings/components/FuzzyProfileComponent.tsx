import { Box, Button, DialogContentText, FormControlLabel, FormGroup, Grid, Switch, TextField } from '@mui/material';
import { Fragment, useEffect, useState } from 'react';
import { useSnackbar } from 'notistack';
import { } from 'chart.js';
import { FuzzyService } from '../../../../modules/fuzzy/fuzzy-service';
import GraphFuzzyDistributionComponent from '../../../../modules/fuzzy/components/GraphFuzzyDistributionComponent';
import { FuzzyProfileDto } from '../../../../modules/fuzzy/dtos/fuzzy-profile-dto';
import { FUZZY_CONSTANTS } from '../../../../modules/fuzzy/fuzzy-constants';
import { FuzzyVariableYear } from '../../../../modules/fuzzy/models/fuzzy-variable-year';
import FuzzyVarComponent from '../../../../modules/fuzzy/components/FuzzyVarComponent';
import { FuzzyVariableDuration } from '../../../../modules/fuzzy/models/fuzzy-variable-duration';
import { FuzzyVariableI } from '../../../../modules/fuzzy/models/fuzzy-variable-distribution';
import { FuzzyVariableRating } from '../../../../modules/fuzzy/models/fuzzy-variable-rating';
import { FuzzyVariablePopularity } from '../../../../modules/fuzzy/models/fuzzy-variable-popularity';
import { FuzzyWeights } from '../../../../modules/fuzzy/models/fuzzy-weights';
import { Add, Edit, Delete } from '@mui/icons-material';
import { FuzzyProfileData } from '../../../../modules/fuzzy/models/fuzzy-profile-data';
import { FuzzySettingsService } from '../fuzzy-settings-service';
import ConfirmationDialogComponent from '../../../../modules/ui/components/MovieDialogDeleteComponent';
import { ConcreteWeights } from '../../../../modules/fuzzy/models/concrete-weights';
import ConcreteWeightsComponent from './ConcreteWeightsComponent';

export interface FuzzyProfileComponentProps {
    fuzzyProfileDto: FuzzyProfileDto;
    readonly: boolean;
    onProfileChanged?: (name: string) => void;
}

export default function FuzzyProfileComponent({ fuzzyProfileDto, readonly, onProfileChanged }: FuzzyProfileComponentProps) {
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


    const { enqueueSnackbar } = useSnackbar();

    // useEffect(() => {
    //     setFuzzyVariableYear(FuzzyService.convertFuzzyVariableTo(fuzzyProfileDto.fuzzyProfileData.fuzzyVariableYear));
    //     setFuzzyVariableRating(FuzzyService.convertFuzzyVariableTo(fuzzyProfileDto.fuzzyProfileData.fuzzyVariableRating));
    //     setFuzzyVariablePopularity(FuzzyService.convertFuzzyVariableTo(fuzzyProfileDto.fuzzyProfileData.fuzzyVariablePopularity));
    // }, [])

    function stateToFuzzyProfileDto(): FuzzyProfileDto {
        return new FuzzyProfileDto({
            fuzzyProfileId: null,
            name: name,
            fuzzyProfileData: new FuzzyProfileData({
                fuzzyVariableYear,
                fuzzyVariableRating,
                fuzzyVariablePopularity,
                fuzzyVariableDuration,
                fuzzyWeights,
                concreteWeights,
            }),
            showTopsisAnalysis: showTopsisAnalysis,
            active,
            useFuzzyTopsis
        })
    }

    async function handleCreateProfile() {
        const newFuzzyProfileDto = stateToFuzzyProfileDto();
        console.log('newFuzzyProfileDto', newFuzzyProfileDto)
        try {
            const response = await FuzzySettingsService.createFuzzyProfile(newFuzzyProfileDto);
            enqueueSnackbar('Επιτυχημένη δημιουργία Fuzzy Profile', { variant: 'success' })
            if (onProfileChanged) {
                onProfileChanged(name);
            }
            // props.afterAdd(e);
        } catch (e: any) {
            if (e?.response?.status === 422) {
                console.error(e?.response?.data?.error);
                enqueueSnackbar('Αποτυχημένη δημιουργία Fuzzy Profile: ' + e?.response?.data?.error, { variant: 'error' })
            } else {
                console.error(e);
                enqueueSnackbar('Αποτυχημένη δημιουργία Fuzzy Profile', { variant: 'error' })
            }
            // if (e.response.statu)
        }

    }

    async function handleUpdateProfile() {
        const updatedFuzzyProfileDto = stateToFuzzyProfileDto();
        updatedFuzzyProfileDto.fuzzyProfileId = fuzzyProfileDto.fuzzyProfileId;
        console.log('newFuzzyProfileDto', updatedFuzzyProfileDto)
        try {
            const response = await FuzzySettingsService.updateFuzzyProfile(updatedFuzzyProfileDto);
            enqueueSnackbar('Επιτυχημένη αποθήκευση Fuzzy Profile', { variant: 'success' })
            if (onProfileChanged) {
                onProfileChanged(name);
            }
            // props.afterAdd(e);
        } catch (e: any) {
            if (e?.response?.status === 422) {
                console.error(e?.response?.data?.error);
                enqueueSnackbar('Αποτυχημένη αποθήκευση Fuzzy Profile: ' + e?.response?.data?.error, { variant: 'error' })
            } else {
                console.error(e);
                enqueueSnackbar('Αποτυχημένη αποθήκευση Fuzzy Profile', { variant: 'error' })
            }
            // if (e.response.statu)
        }
    }

    async function onDeleteProfileOk() {
        try {
            const response = await FuzzySettingsService.deleteFuzzyProfile(name);
            enqueueSnackbar('Επιτυχημένη διαγραφή Fuzzy Profile', { variant: 'success' })
            if (onProfileChanged) {
                onProfileChanged(FUZZY_CONSTANTS.DEFAULT);
            }
            // props.afterAdd(e);
        } catch (e: any) {
            if (e?.response?.status === 422) {
                console.error(e?.response?.data?.error);
                enqueueSnackbar('Αποτυχημένη διαγραφή Fuzzy Profile: ' + e?.response?.data?.error, { variant: 'error' })
            } else {
                console.error(e);
                enqueueSnackbar('Αποτυχημένη διαγραφή Fuzzy Profile', { variant: 'error' })
            }
            // if (e.response.statu)
        }
        setDeleteProfileConfirmationDialogOpen(false);
    }
    async function handleDeleteProfile() {
        setDeleteProfileConfirmationDialogOpen(true);
    }

    function handleShowTopsisAnalysisChange(event: React.ChangeEvent<HTMLInputElement>) {
        setShowTopsisAnalysis(event.target.checked);
    }
    function handleActiveChange(event: React.ChangeEvent<HTMLInputElement>) {
        setActive(event.target.checked);
    }
    function handleUseFuzzyTopsisChange(event: React.ChangeEvent<HTMLInputElement>) {
        setUseFuzzyTopsis(event.target.checked);
    }

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
                    <Grid item>
                        {fuzzyProfileDto.name === '' && (
                            <Button onClick={handleCreateProfile} variant="contained" startIcon={<Add />}>
                                ΔΗΜΙΟΥΡΓΙΑ
                            </Button>
                        )}
                        {fuzzyProfileDto.name !== '' && fuzzyProfileDto.name !== FUZZY_CONSTANTS.DEFAULT && (
                            <Fragment>
                                <Button onClick={handleUpdateProfile} variant="contained" startIcon={<Edit />}>
                                    ΑΠΟΘΗΚΕΥΣΗ
                                </Button>
                                <Button sx={{ marginLeft: 1 }} onClick={handleDeleteProfile} variant="contained" startIcon={<Delete />}>
                                    ΔΙΑΓΡΑΦΗ
                                </Button>
                            </Fragment>
                        )}
                    </Grid>
                </Grid>
                <FormGroup>
                    <FormControlLabel disabled={readonly} control={<Switch onChange={handleShowTopsisAnalysisChange} />} label="ShowTopsisAnalysis" checked={showTopsisAnalysis} />
                    <FormControlLabel disabled={readonly} control={<Switch onChange={handleActiveChange} />} label="Active" checked={active} />
                    <FormControlLabel disabled={readonly} control={<Switch onChange={handleUseFuzzyTopsisChange} />} label="UseFuzzyTopsis" checked={useFuzzyTopsis} />
                </FormGroup>
                <hr></hr>

                <FuzzyVarComponent fuzzyVariable={fuzzyVariableYear} readonly={readonly} xStepSize={5}></FuzzyVarComponent>
                <hr></hr>
                <br></br>

                <FuzzyVarComponent fuzzyVariable={fuzzyVariableRating} readonly={readonly} xStepSize={1}></FuzzyVarComponent>
                <hr></hr>
                <br></br>

                <FuzzyVarComponent fuzzyVariable={fuzzyVariablePopularity} readonly={readonly} xStepSize={25}></FuzzyVarComponent>
                <hr></hr>
                <br></br>

                <FuzzyVarComponent fuzzyVariable={fuzzyVariableDuration} readonly={readonly} xStepSize={20}></FuzzyVarComponent>
                <hr></hr>
                <br></br>

                <FuzzyVarComponent fuzzyVariable={fuzzyWeights} readonly={readonly} xStepSize={0.1} numberStep={0.1}></FuzzyVarComponent>
                <ConcreteWeightsComponent concreteWeights={concreteWeights} readonly={readonly}></ConcreteWeightsComponent>

            </Box>
            <ConfirmationDialogComponent open={deleteProfileConfirmationDialogOpen} onOk={onDeleteProfileOk} onCancel={() => setDeleteProfileConfirmationDialogOpen(false)}>
                <DialogContentText>
                    Είστε σίγουρος ότι θέλετε να διαγράψεται αυτό το Fuzzy Profile?
                </DialogContentText>
            </ConfirmationDialogComponent>
        </Fragment>
    );
}
