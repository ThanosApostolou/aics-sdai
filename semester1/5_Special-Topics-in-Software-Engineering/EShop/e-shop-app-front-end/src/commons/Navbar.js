import UserService from "../services/UserService";
import '../App.css'
import { useEffect, useState } from 'react';
import ExitToApp from '@mui/icons-material/ExitToApp';
import Button from '@mui/material/Button';
import { styled } from '@mui/material/styles';
import { grey } from '@mui/material/colors';

const LogoutButton = styled(Button)({
  backgroundColor: grey[900],
  color: '#fff',
  '&:hover': {
    backgroundColor: grey[700],
  },
});

function Navbar() {
  const [userName, setUserName] = useState('');
  const [role, setRole] = useState('');

  useEffect(() => {
    setUserName(UserService.getUsername());
    setRole(UserService.getRole());
  }, []);

  useEffect(() => {
    console.log('userName', userName);
    console.log('role', role);
  }, [userName,role]);

  const handleLogout = () => {
    if (window.confirm('Are you sure you want to logout?')) {
      UserService.doLogout();
    }
  };

  return (

    <div className="nav-bar-container-light">
      <div className='alignLeft'>
        <h2>Welcome,</h2>
        <h2>{role} : {userName}!</h2>
      </div>

      <div className='alignRight'>
        <LogoutButton
          variant="contained"
          startIcon={<ExitToApp />}
          onClick={handleLogout}
        >
          Logout
        </LogoutButton>
      </div>
    </div>
  );
}

export default Navbar