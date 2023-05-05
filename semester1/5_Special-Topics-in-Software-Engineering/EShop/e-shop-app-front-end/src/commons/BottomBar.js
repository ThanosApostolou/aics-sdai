import { BottomNavigation, BottomNavigationAction } from '@mui/material';
import { Link } from 'react-router-dom'
import Home from '@mui/icons-material/Home';
import Info from '@mui/icons-material/Info';
import Room from '@mui/icons-material/Room';
import '../App.css'

function BottomBar() {
    return (
            <BottomNavigation>
                <BottomNavigationAction
                    component={Link}
                    to="/home"
                    label="Home"
                    value="Home"
                    icon={<Home />}
                >
                </BottomNavigationAction>
                <BottomNavigationAction
                    component={Link}
                    to="/location"
                    label="Location"
                    value="Location"
                    icon={<Room />}
                >
                </BottomNavigationAction>
                <BottomNavigationAction
                    component={Link}
                    to="/about"
                    label="About"
                    value="About"
                    icon={<Info />}
                >
                </BottomNavigationAction>
            </BottomNavigation>
            
    );
}

export default BottomBar