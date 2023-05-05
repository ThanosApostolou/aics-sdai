import * as React from 'react';
import Box from '@mui/material/Box';
import Modal from '@mui/material/Modal';
import { useState } from 'react';
import AddPayment from "./AddPayment";
import "../../App.css";

const style = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};

export default function BasicModal(props) {
  const [open, setOpen] = useState(props.open);
  const handleClose = () => setOpen(false);

  return (
    < div >
      <button onClick={props.handleOpen} className="buttonInsert">+ Add New Payment</button>
      <Modal
        open={props.open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box sx={style}>
          <div className='alignDivRight'><button onClick={props.handleOpen} className="buttonDelete">x</button></div>
          <br></br><br></br>
          <AddPayment {...props} addPaymenttHandler={props.addPaymenttHandler} />
        </Box>
      </Modal>
    </div >
  );
}