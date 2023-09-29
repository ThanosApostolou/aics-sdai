import Button from '@mui/material/Button';
import { Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from '@mui/material';
import { PropsWithChildren } from 'react';

export interface ConfirmationDialogComponentProps {
    open: boolean;
    title?: string;
    okText?: string;
    cancelText?: string;
    onOk?: (event: any) => void;
    onCancel?: (event: any) => void;
}

export default function ConfirmationDialogComponent({ open, title, okText, cancelText, children, onOk, onCancel }: PropsWithChildren<ConfirmationDialogComponentProps>) {
    return (
        <Dialog fullWidth={true} maxWidth={false} onClose={onCancel} open={open}>
            <DialogTitle id="alert-dialog-title">
                {title ? title : 'Confirmation'}
            </DialogTitle>
            <DialogContent>
                {children}
            </DialogContent>
            <DialogActions>
                <Button onClick={onCancel}>{okText ? okText : 'Ακύρωση'}</Button>
                <Button onClick={onOk} autoFocus>{cancelText ? cancelText : 'Οκ'}</Button>
            </DialogActions>

        </Dialog>
    )
}
