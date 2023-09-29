import { Box, Button, FormControl, Grid, InputLabel, MenuItem, Select, SelectChangeEvent, TextField } from '@mui/material';
import { Fragment, useEffect, useMemo, useState } from 'react';
import { useSnackbar } from 'notistack';
import { ChartData } from 'chart.js';
import { FuzzyService } from '../fuzzy-service';
import GraphFuzzyDistributionComponent from './GraphFuzzyDistributionComponent';
import { FuzzyProfileDto } from '../dtos/fuzzy-profile-dto';
import { FUZZY_CONSTANTS, FuzzyVariableDistributionType, FuzzyVariablePartPosition } from '../fuzzy-constants';
import { FuzzyVariableDistributionPart, FuzzyVariableDistributionPartTrapezoidal, FuzzyVariableDistributionPartTriangular, FuzzyVariableI } from '../models/fuzzy-variable-distribution';
import MyKatexComponent from '../../mykatex/MyKatexComponent';

export interface FuzzyVarPartComponentProps {
    fuzzyVariableDistributionPart: FuzzyVariableDistributionPart;
    readonly: boolean;
    fuzzyVariablePartPosition: FuzzyVariablePartPosition;
    numberStep?: number;
    fuzzyVarPartUpdated?: (fuzzyVariableDistributionPart: FuzzyVariableDistributionPart) => void
}

export default function FuzzyVarPartComponent({ fuzzyVariableDistributionPart, readonly, fuzzyVariablePartPosition, numberStep, fuzzyVarPartUpdated }: FuzzyVarPartComponentProps) {
    const [type, setType] = useState<FuzzyVariableDistributionType>(fuzzyVariableDistributionPart.type);
    const [a, setA] = useState<number | null>(fuzzyVariableDistributionPart.a);
    const [b, setB] = useState<number>(fuzzyVariableDistributionPart.b);
    const [c, setC] = useState<number | null>(fuzzyVariableDistributionPart.c);
    const [d, setD] = useState<number | null>(fuzzyVariableDistributionPart.isTypeTrapezoidal() ? fuzzyVariableDistributionPart.d : null);
    const { enqueueSnackbar } = useSnackbar();
    const latexStr = useMemo(() => createLatexEquals(), [a, b, c, d]);

    useEffect(() => {
        if (fuzzyVarPartUpdated) {
            fuzzyVarPartUpdated(stateToFuzzyVariableDistributionPart())
        }
    }, [type, a, b, c, d])

    function stateToFuzzyVariableDistributionPart(): FuzzyVariableDistributionPart {
        if (FuzzyVariableDistributionType.TRIANGULAR === type) {
            return new FuzzyVariableDistributionPartTriangular({
                partName: fuzzyVariableDistributionPart.partName,
                a: a,
                b: b,
                c: c,
            })
        } else {
            return new FuzzyVariableDistributionPartTrapezoidal({
                partName: fuzzyVariableDistributionPart.partName,
                a: a,
                b: b,
                c: c != null ? c : 0,
                d: d
            })
        }

    }

    // function stateToFuzzyVariableI(): FuzzyVariableI {
    //     // return fuzzyVariable.getFuzzyVariableMap;
    //     return fuzzyVariable;

    // }

    function handleTypeChange(event: SelectChangeEvent<FuzzyVariableDistributionType>) {
        const type = event.target.value as FuzzyVariableDistributionType;
        setType(type);
        if (FuzzyVariablePartPosition.END !== fuzzyVariablePartPosition) {
            if (FuzzyVariableDistributionType.TRIANGULAR === type) {
                setD(null);
            } else if (FuzzyVariableDistributionType.TRAPEZOIDAL === type) {
                setD(c);
            }
        } else {
            if (FuzzyVariableDistributionType.TRIANGULAR === type) {
                setC(null);
                setD(null);
            } else if (FuzzyVariableDistributionType.TRAPEZOIDAL === type) {
                setC(b);
                setD(null);
            }
        }
    }

    function createLatexEquals(): string {
        const partName = fuzzyVariableDistributionPart.partName.replace('_', '\\_');

        let branches = '';
        if (type === FuzzyVariableDistributionType.TRIANGULAR) {
            if (a != null && c != null) {
                branches += String.raw`0 & \text{, } x \leq ${a} \text{ or } x \geq ${c} \\\\`
            } else if (a != null) {
                branches += String.raw`0 & \text{, } x \leq ${a} \\\\`
            } else if (c != null) {
                branches += String.raw`0 & \text{, } x \geq ${c} \\\\`
            }
            if (a != null) {
                branches += String.raw`{x - ${a}} \over {${(b - a).toFixed(2)}} & \text{, } ${a} \leq x \leq ${b} \\`
            }
            if (c != null) {
                branches += String.raw`\\ {${c} - x} \over {${(c - b).toFixed(2)}} & \text{, } ${b} \leq x \leq ${c} \\`
            }
        } else if (type === FuzzyVariableDistributionType.TRAPEZOIDAL) {
            if (a != null && d != null) {
                branches += String.raw`0 & \text{, } x \leq ${a} \text{ or } x \geq ${d} \\\\`
            } else if (a != null) {
                branches += String.raw`0 & \text{, } x \leq ${a} \\\\`
            } else if (d != null) {
                branches += String.raw`0 & \text{, } x \geq ${d} \\\\`
            }
            if (a != null) {
                branches += String.raw`{x - ${a}} \over {${(b - a).toFixed(2)}} & \text{, } ${a} \leq x \leq ${b} \\\\`
            }
            branches += String.raw`1 & \text{, } ${b} \leq x \leq ${c} \\`
            if (c != null && d != null) {
                branches += String.raw`\\ {${d} - x} \over {${(d - c).toFixed(2)}} & \text{, } ${c} \leq x \leq ${d} \\`
            }
        }
        return String.raw`μ_{${partName}}(x) =
        \left\{
            \begin{array}{ll}
                ${branches}
            \end{array}
        \right.`
    }

    return (
        <Fragment>
            <Grid container spacing={2} sx={{ justifyContent: 'center', justifyItems: 'center' }}>
                <h5 style={{ textAlign: 'center' }}>{fuzzyVariableDistributionPart.partName}</h5>
            </Grid>

            <Grid container spacing={2} sx={{ padding: 1 }}>

                <Grid item xs={12} sm={4}>
                    <FormControl disabled={readonly} size='small'>
                        <InputLabel id="label-select-type">Type</InputLabel>
                        <Select
                            labelId="label-select-type"
                            id="select-type"
                            value={type}
                            label="Type"
                            onChange={handleTypeChange}
                        >
                            <MenuItem value={FuzzyVariableDistributionType.TRIANGULAR}>{FuzzyVariableDistributionType.TRIANGULAR}</MenuItem>
                            <MenuItem value={FuzzyVariableDistributionType.TRAPEZOIDAL}>{FuzzyVariableDistributionType.TRAPEZOIDAL}</MenuItem>
                        </Select>

                    </FormControl>

                    {fuzzyVariablePartPosition !== FuzzyVariablePartPosition.START && (
                        <TextField size='small' sx={{ width: '6rem' }}
                            disabled={readonly} type="number" inputProps={{ step: numberStep != null ? numberStep : 1 }} label="a" value={a != null ? a : ''} onChange={(e) => setA(e.target.value ? parseFloat(e.target.value) : null)} />
                    )}
                    <TextField size='small' sx={{ width: '6rem' }}
                        disabled={readonly} type="number" inputProps={{ step: numberStep != null ? numberStep : 1 }} label="b" value={b} onChange={(e) => setB(e.target.value ? parseFloat(e.target.value) : 0)} />
                    {!(type === FuzzyVariableDistributionType.TRIANGULAR && fuzzyVariablePartPosition === FuzzyVariablePartPosition.END) && (
                        <TextField size='small' sx={{ width: '6rem' }}
                            disabled={readonly} type="number" inputProps={{ step: numberStep != null ? numberStep : 1 }} label="c" value={c != null ? c : ''} onChange={(e) => setC(e.target.value ? parseFloat(e.target.value) : null)} />

                    )}
                    {type === FuzzyVariableDistributionType.TRAPEZOIDAL && fuzzyVariablePartPosition !== FuzzyVariablePartPosition.END && (
                        <TextField size='small' sx={{ width: '6rem' }}
                            disabled={readonly} type="number" inputProps={{ step: numberStep != null ? numberStep : 1 }} label="d" value={d != null ? d : ''} onChange={(e) => setD(e.target.value ? parseFloat(e.target.value) : null)} />
                    )}

                </Grid>
                <Grid item xs={12} sm={8}>
                    <MyKatexComponent latexStr={latexStr}></MyKatexComponent>
                </Grid>

            </Grid>
        </Fragment>
    );
}
