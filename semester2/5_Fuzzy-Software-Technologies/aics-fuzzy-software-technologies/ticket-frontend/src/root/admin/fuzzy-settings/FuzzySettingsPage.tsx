import { Box, Button, CircularProgress, FormControl, Grid, InputLabel, MenuItem, Select, SelectChangeEvent } from '@mui/material';
import { Fragment, useEffect, useRef, useState } from 'react';
import { FuzzySettingsService } from './fuzzy-settings-service';
import { useSnackbar } from 'notistack';
import { FUZZY_CONSTANTS } from '../../../modules/fuzzy/fuzzy-constants';
import { FetchFuzzyProfilesResponseDto } from './dtos/fetch-fuzzy-profiles-dto';
import FuzzyProfileComponent from './components/FuzzyProfileComponent';
import React from 'react';
import { FuzzyProfileDto } from '../../../modules/fuzzy/dtos/fuzzy-profile-dto';
import { Add, Delete, Edit } from '@mui/icons-material';
export default function FuzzySettingsPage() {
    const [fetchFuzzyProfilesResponseDto, setFetchFuzzyProfilesResponseDto] = useState<FetchFuzzyProfilesResponseDto | null>(null)
    const [fuzzyProfileDto, setFuzzyProfileDto] = useState<FuzzyProfileDto | null>(null);
    const [selectedProfileName, setSelectedProfileName] = useState<string>(FUZZY_CONSTANTS.DEFAULT);

    const { enqueueSnackbar } = useSnackbar();

    useEffect(() => {
        loadData(selectedProfileName);
    }, [])

    async function loadData(selectedProfileName: string) {
        setFetchFuzzyProfilesResponseDto(null);
        try {
            setFetchFuzzyProfilesResponseDto(null);
            setFuzzyProfileDto(null);
            const fetchFuzzyProfilesResponseDto = await FuzzySettingsService.fetchFuzzyProfiles();
            console.log('fetchFuzzyProfilesResponseDto', fetchFuzzyProfilesResponseDto)
            console.log('selectedProfileName', selectedProfileName)
            const newFuzzyProfileDto: FuzzyProfileDto = fetchFuzzyProfilesResponseDto.fuzzyProfilesMap[selectedProfileName];
            setFetchFuzzyProfilesResponseDto(fetchFuzzyProfilesResponseDto);
            setFuzzyProfileDto(newFuzzyProfileDto.deepClone());
        } catch (e) {
            console.error(e);
            enqueueSnackbar('Αποτυχημένη εύρεση των Fuzzy Profiles', { variant: 'error' })
        }
    }

    function handleProfileChange(event: SelectChangeEvent<string>) {
        const newProfileName = event.target.value;
        if (FUZZY_CONSTANTS.NEW === newProfileName) {
            const defaultProfile = fetchFuzzyProfilesResponseDto?.fuzzyProfilesMap[FUZZY_CONSTANTS.DEFAULT];
            if (!defaultProfile) {
                console.error(`cannot find DEFAULT profile`);
                return;
            }
            const newProfile = defaultProfile.deepClone()
            newProfile.name = '';
            setSelectedProfileName(FUZZY_CONSTANTS.NEW)
            setFuzzyProfileDto(null)
            setTimeout(() => {
                setFuzzyProfileDto(newProfile);
            })
        } else {
            const newProfile = fetchFuzzyProfilesResponseDto?.fuzzyProfilesMap[newProfileName];
            if (!newProfile) {
                console.error(`cannot find ${newProfileName}`);
                return;
            }
            setSelectedProfileName(newProfileName)
            setFuzzyProfileDto(null)
            setTimeout(() => {
                setFuzzyProfileDto(newProfile.deepClone());
            })
        }
    }

    async function handleProfileChanged(name: string) {
        console.log('name', name)
        await loadData(name);
        setSelectedProfileName(name);
    }

    return (
        <Fragment>
            <Box style={{ width: '100%', height: '100%' }}>
                <h1>Fuzzy Settings</h1>

                {fetchFuzzyProfilesResponseDto && fuzzyProfileDto
                    ? (
                        <React.Fragment>
                            <FormControl>
                                <InputLabel id="label-select-profile">Profile</InputLabel>
                                <Select
                                    labelId="label-select-profile"
                                    id="select-profile"
                                    value={selectedProfileName}
                                    label="Profile"
                                    onChange={handleProfileChange}
                                >
                                    {fetchFuzzyProfilesResponseDto.fuzzyProfilesNames.map(profileName => (
                                        <MenuItem key={profileName} value={profileName}>{profileName}</MenuItem>
                                    ))}
                                    <MenuItem key={FUZZY_CONSTANTS.NEW} value={FUZZY_CONSTANTS.NEW}>{FUZZY_CONSTANTS.NEW}</MenuItem>
                                </Select>
                            </FormControl>

                            {fuzzyProfileDto && (
                                <FuzzyProfileComponent fuzzyProfileDto={fuzzyProfileDto} readonly={fuzzyProfileDto.name === FUZZY_CONSTANTS.DEFAULT}
                                    onProfileChanged={handleProfileChanged}></FuzzyProfileComponent>
                            )}
                        </React.Fragment>
                    )
                    : (
                        <CircularProgress />
                    )}

            </Box>
        </Fragment>
    );
}
