
import { Accordion, AccordionActions, AccordionDetails, AccordionSummary, Box, Button, FormControl, FormControlLabel, Grid, InputLabel, MenuItem, Select, SelectChangeEvent, Stack, Switch, TextField, Typography } from '@mui/material';
import { FuzzySearchFiltersDto } from '../../../../modules/fuzzy/dtos/fuzzy-search-filters-dto';
import { Fragment, useState } from 'react';
import { FuzzySearchChoices } from '../../../../modules/fuzzy/fuzzy-constants';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SearchIcon from '@mui/icons-material/Search';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import { FuzzySearchTopsisAnalysisDto } from '../../../../modules/fuzzy/dtos/fuzzy-search-topsis-analysis-dto';
import AdbIcon from '@mui/icons-material/Adb';
import TopsisAnalysisDialogComponent from './TopsisAnalysisDialogComponent';

export interface FuzzySearchAccordionComponentProps {
    fuzzySearchTopsisAnalysisDto: FuzzySearchTopsisAnalysisDto | null;
    onSearch?: (fuzzySearchFiltersDto: FuzzySearchFiltersDto | null) => void;
}



export default function FuzzySearchAccordionComponent({ fuzzySearchTopsisAnalysisDto, onSearch }: FuzzySearchAccordionComponentProps) {
    const [choice1, setChoice1] = useState<FuzzySearchChoices>(FuzzySearchChoices.RATING);
    const [choice2, setChoice2] = useState<FuzzySearchChoices>(FuzzySearchChoices.POPULARITY);
    const [choice3, setChoice3] = useState<FuzzySearchChoices>(FuzzySearchChoices.YEAR);
    const [choice4, setChoice4] = useState<FuzzySearchChoices>(FuzzySearchChoices.DURATION);
    const [yearCostCriteria, setYearCostCriteria] = useState<boolean>(false);
    const [durationCostCriteria, setDurationCostCriteria] = useState<boolean>(false);
    const [topsisAnalysisDialogOpen, setTopsisAnalysisDialogOpen] = useState<boolean>(false);

    function choice1Updated(e: SelectChangeEvent<FuzzySearchChoices>) {
        const newChoice1 = e.target.value as FuzzySearchChoices;
        setChoice1(newChoice1)
    }
    function choice2Updated(e: SelectChangeEvent<FuzzySearchChoices>) {
        const newChoice2 = e.target.value as FuzzySearchChoices;
        setChoice2(newChoice2)
    }
    function choice3Updated(e: SelectChangeEvent<FuzzySearchChoices>) {
        const newChoice3 = e.target.value as FuzzySearchChoices;
        setChoice3(newChoice3)
    }
    function choice4Updated(e: SelectChangeEvent<FuzzySearchChoices>) {
        const newChoice4 = e.target.value as FuzzySearchChoices;
        setChoice4(newChoice4)
    }

    function handleClear() {
        setChoice1(FuzzySearchChoices.RATING);
        setChoice2(FuzzySearchChoices.POPULARITY);
        setChoice3(FuzzySearchChoices.YEAR);
        setChoice4(FuzzySearchChoices.DURATION);
        setYearCostCriteria(false);
        setDurationCostCriteria(false);
        if (onSearch) {
            onSearch(null);
        }
    }

    function handleSearch() {
        const fuzzySearchFiltersDto = new FuzzySearchFiltersDto({
            choice1,
            choice2,
            choice3,
            choice4,
            yearCostCriteria,
            durationCostCriteria
        });
        if (onSearch) {
            onSearch(fuzzySearchFiltersDto);
        }
    }

    function handleTopsisAnalysis() {
        setTopsisAnalysisDialogOpen(true);
    }

    return (
        <Box style={{ width: '100%', height: '100%', padding: 2 }}>
            <Accordion>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls="panel1a-content"
                    id="panel1a-header"
                >
                    <h2>Fuzzy Search</h2>
                </AccordionSummary>
                <AccordionDetails>
                    <p>
                        Επιλέξτε όλα τα κριτήρια με την σειρά από το πιο σημαντικό για εσάς ως το λιγότερο σημαντικό για να σας προτείνουμε ταινίες!
                    </p>
                    <Stack direction='row' spacing={2} sx={{ padding: 1 }}>
                        <div>
                            <FormControlLabel disabled={false} control={<Switch onChange={(e) => setYearCostCriteria(e.target.checked)} />} label="Προτίμηση παλιότερων ταινιών?" checked={yearCostCriteria} />
                        </div>
                        <div>
                            <FormControlLabel disabled={false} control={<Switch onChange={(e) => setDurationCostCriteria(e.target.checked)} />} label="Προτίμηση μικρότερων ταινιών?" checked={durationCostCriteria} />
                        </div>
                        <div>
                            <FormControl size='small'>
                                <InputLabel id="label-select-choice1">1st Choice</InputLabel>
                                <Select
                                    labelId="label-select-choice1"
                                    id="select-choice1"
                                    value={choice1}
                                    label="1st Choice"
                                    onChange={choice1Updated}
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
                                    value={choice2}
                                    label="2nd Choice"
                                    onChange={choice2Updated}
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
                                    value={choice3}
                                    label="3rd Choice"
                                    onChange={choice3Updated}
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
                                    value={choice4}
                                    label="4th Choice"
                                    onChange={choice4Updated}
                                >
                                    {Object.values(FuzzySearchChoices).map(choice => (
                                        <MenuItem key={choice} value={choice}>{choice}</MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </div>
                    </Stack>
                </AccordionDetails>
                <AccordionActions>
                    {fuzzySearchTopsisAnalysisDto != null && (
                        <Fragment>
                            <Button startIcon={<AdbIcon />} color='info' onClick={handleTopsisAnalysis}>TOPSIS Analysis</Button>
                            <TopsisAnalysisDialogComponent open={topsisAnalysisDialogOpen} fuzzySearchTopsisAnalysisDto={fuzzySearchTopsisAnalysisDto} onClose={() => setTopsisAnalysisDialogOpen(false)}></TopsisAnalysisDialogComponent>
                        </Fragment>
                    )}
                    <Button startIcon={<RestartAltIcon />} color='warning' onClick={handleClear}>Επαναφορά</Button>
                    <Button startIcon={<SearchIcon />} onClick={handleSearch}>Αναζήτηση</Button>
                </AccordionActions>
            </Accordion>
        </Box>
    )
}
