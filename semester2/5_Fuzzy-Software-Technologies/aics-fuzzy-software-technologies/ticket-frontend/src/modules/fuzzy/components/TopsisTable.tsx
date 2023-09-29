
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import { Box } from '@mui/material';
import { useEffect, useMemo, useState } from 'react';
import { TopsisDataRowDto } from '../dtos/topsis-data-row-dto';
import { TopsisDataTableDto } from '../dtos/topsis-data-table-dto';

import './TopsisTable.css';

export interface TopsisTableProps {
    dataTable: TopsisDataTableDto;
    yearCostCriteria: boolean;
    durationCostCriteria: boolean;
}

export default function TopsisTable({ dataTable, yearCostCriteria, durationCostCriteria }: TopsisTableProps) {
    const columns = useMemo(() => {
        return createColumns(dataTable.showDpos, dataTable.showDneg, dataTable.showScore)
    }, [dataTable]);

    function createColumns(showDpos: boolean, showDneg: boolean, showScore: boolean): GridColDef[] {
        const columns: GridColDef[] = [
            {
                field: 'name',
                headerName: 'Movies Alternatives',
                minWidth: 100,
                flex: 1,
                editable: false,
                sortable: false
            },
            {
                field: 'rating',
                headerName: 'Rating (B)',
                minWidth: 50,
                flex: 1,
                editable: false,
                sortable: false,
                headerClassName: 'header-benefit',
            }, {
                field: 'popularity',
                headerName: 'Popularity (C)',
                width: 50,
                flex: 1,
                editable: false,
                sortable: false,
                headerClassName: 'header-cost',
            }, {
                field: 'year',
                headerName: `Year ${yearCostCriteria ? '(C)' : '(B)'}`,
                width: 50,
                flex: 1,
                editable: false,
                sortable: false,
                headerClassName: `${yearCostCriteria ? 'header-cost' : 'header-benefit'}`,
            }, {
                field: 'duration',
                headerName: `Duration ${durationCostCriteria ? '(C)' : '(B)'}`,
                width: 50,
                flex: 1,
                editable: false,
                sortable: false,
                headerClassName: `${durationCostCriteria ? 'header-cost' : 'header-benefit'}`,
            }
        ];
        if (showDpos) {
            columns.push({
                field: 'dpos',
                headerName: 'D+',
                width: 50,
                flex: 1,
                editable: false,
                sortable: false
            })
        }
        if (showDneg) {
            columns.push({
                field: 'dneg',
                headerName: 'D-',
                width: 50,
                flex: 1,
                editable: false,
                sortable: false
            })
        }
        if (showScore) {
            columns.push({
                field: 'score',
                headerName: 'Score',
                width: 50,
                flex: 1,
                editable: false,
                sortable: false
            });
        }
        return columns;
    }

    return (
        <Box sx={{ height: '400px', padding: 2 }}>
            <DataGrid
                rows={dataTable.rows}
                columns={columns}
                hideFooter={true}
                checkboxSelection={false}
                disableRowSelectionOnClick
                getRowId={(row) => row.movieId}
                density='compact'
                disableColumnMenu
            />
        </Box>
    )
}
