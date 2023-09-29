
import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack, Switch, Tab, Tabs } from '@mui/material';
import { FuzzySearchFiltersDto } from '../../../../modules/fuzzy/dtos/fuzzy-search-filters-dto';
import { useState } from 'react';
import { FuzzySearchChoices } from '../../../../modules/fuzzy/fuzzy-constants';
import { FuzzySearchTopsisAnalysisDto } from '../../../../modules/fuzzy/dtos/fuzzy-search-topsis-analysis-dto';
import { FuzzyProfileDto } from '../../../../modules/fuzzy/dtos/fuzzy-profile-dto';
import TabPanel from '../../../../modules/ui/components/TabPanelComponent';
import TabFuzzyProfileComponent from './TabFuzzyProfileComponent';
import TabRegularTopsisComponent from './TabRegularTopsisComponent';
import { RegularTopsisInfoDto } from '../../../../modules/fuzzy/dtos/regular-topsis-info-dto';
import { FuzzyTopsisInfoDto } from '../../../../modules/fuzzy/dtos/fuzzy-topsis-info-dto';
import TabFuzzyTopsisComponent from './TabFuzzyTopsisComponent';

export interface TopsisAnalysisDialogComponentProps {
    fuzzySearchTopsisAnalysisDto: FuzzySearchTopsisAnalysisDto;
    open: boolean;
    onClose?: () => void;
}



export default function TopsisAnalysisDialogComponent({ fuzzySearchTopsisAnalysisDto, open, onClose }: TopsisAnalysisDialogComponentProps) {
    const [fuzzyProfileDto, setFuzzyProfileDto] = useState<FuzzyProfileDto>(fuzzySearchTopsisAnalysisDto.fuzzyProfileDto);
    const [fuzzySearchFiltersDto, setFuzzySearchFiltersDto] = useState<FuzzySearchFiltersDto>(fuzzySearchTopsisAnalysisDto.fuzzySearchFiltersDto);
    const [regularTopsisInfoDto, setRegularTopsisInfoDto] = useState<RegularTopsisInfoDto | null>(fuzzySearchTopsisAnalysisDto.regularTopsisInfoDto);
    const [fuzzyTopsisInfoDto, setFuzzyTopsisInfoDto] = useState<FuzzyTopsisInfoDto | null>(fuzzySearchTopsisAnalysisDto.fuzzyTopsisInfoDto);
    const [tabValue, setTabValue] = useState(0);


    function a11yProps(index: number) {
        return {
            id: `simple-tab-${index}`,
            'aria-controls': `simple-tabpanel-${index}`,
        };
    }

    function handleChange(event: React.SyntheticEvent, newValue: number) {
        setTabValue(newValue);
    }

    return (
        <Dialog fullWidth={true} maxWidth={false} onClose={onClose} open={open}>
            <DialogTitle id="alert-dialog-title">
                TOPSIS Analysis Information
            </DialogTitle>
            <DialogContent>
                <Stack direction='row' spacing={2} sx={{ padding: 1 }}>
                    <div>
                        <FormControlLabel disabled={true} control={<Switch />} label="Προτίμηση παλιότερων ταινιών?" checked={fuzzySearchFiltersDto.yearCostCriteria} />
                    </div>
                    <div>
                        <FormControlLabel disabled={true} control={<Switch />} label="Προτίμηση μικρότερων ταινιών?" checked={fuzzySearchFiltersDto.durationCostCriteria} />
                    </div>
                    <div>
                        <FormControl size='small'>
                            <InputLabel id="label-select-choice1">1st Choice</InputLabel>
                            <Select
                                labelId="label-select-choice1"
                                id="select-choice1"
                                value={fuzzySearchFiltersDto.choice1}
                                label="1st Choice"
                                disabled={true}
                            >
                                {Object.values(FuzzySearchChoices).map(choice => (
                                    <MenuItem key={choice} value={choice}>{choice}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </div>
                    <div>
                        <FormControl size='small'>
                            <InputLabel id="label-select-choice2">2nd Choice</InputLabel>
                            <Select
                                labelId="label-select-choice2"
                                id="select-choice2"
                                value={fuzzySearchFiltersDto.choice2}
                                label="2nd Choice"
                                disabled={true}
                            >
                                {Object.values(FuzzySearchChoices).map(choice => (
                                    <MenuItem key={choice} value={choice}>{choice}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </div>
                    <div>
                        <FormControl size='small'>
                            <InputLabel id="label-select-choice3">3rd Choice</InputLabel>
                            <Select
                                labelId="label-select-choice3"
                                id="select-choice3"
                                value={fuzzySearchFiltersDto.choice3}
                                label="3rd Choice"
                                disabled={true}
                            >
                                {Object.values(FuzzySearchChoices).map(choice => (
                                    <MenuItem key={choice} value={choice}>{choice}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </div>
                    <div>
                        <FormControl size='small'>
                            <InputLabel id="label-select-choice4">4th Choice</InputLabel>
                            <Select
                                labelId="label-select-choice4"
                                id="select-choice4"
                                value={fuzzySearchFiltersDto.choice4}
                                label="4th Choice"
                                disabled={true}
                            >
                                {Object.values(FuzzySearchChoices).map(choice => (
                                    <MenuItem key={choice} value={choice}>{choice}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </div>
                </Stack>
                <Box sx={{ width: '100%' }}>
                    <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                        <Tabs value={tabValue} onChange={handleChange} aria-label="basic tabs example">
                            <Tab label="Fuzzy Profile Used" {...a11yProps(0)} />
                            <Tab label={`${fuzzyTopsisInfoDto != null ? 'FUZZY TOPSIS' : 'TOPSIS'}`} {...a11yProps(1)} />
                        </Tabs>
                    </Box>
                    <TabPanel value={tabValue} index={0}>
                        <TabFuzzyProfileComponent fuzzyProfileDto={fuzzyProfileDto} />
                    </TabPanel>
                    <TabPanel value={tabValue} index={1}>
                        {regularTopsisInfoDto != null && (
                            <TabRegularTopsisComponent fuzzyProfileDto={fuzzyProfileDto} fuzzySearchFiltersDto={fuzzySearchFiltersDto} regularTopsisInfoDto={regularTopsisInfoDto} />
                        )}
                        {fuzzyTopsisInfoDto != null && (
                            <TabFuzzyTopsisComponent fuzzyProfileDto={fuzzyProfileDto} fuzzySearchFiltersDto={fuzzySearchFiltersDto} fuzzyTopsisInfoDto={fuzzyTopsisInfoDto} />
                        )}
                    </TabPanel>
                </Box>

            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>

        </Dialog>
    )
}
