import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import HomeIcon from '@mui/icons-material/Home';
import ShopIcon from '@mui/icons-material/Shop';
import CategoryIcon from '@mui/icons-material/Category';
import PersonIcon from '@mui/icons-material/Person';
import RoleIcon from '@mui/icons-material/SupervisedUserCircle';
import AdminIcon from '@mui/icons-material/AccountCircle';
import CartIcon from '@mui/icons-material/ShoppingCart';
import PaymentIcon from '@mui/icons-material/Payment';
import ReviewIcon from '@mui/icons-material/Star';
import UserService from '../services/UserService';
import '../App.css';

const menuItemsSuperAdmin = [{
  label: 'Home',
  icon: <HomeIcon />,
  path: '/'
},
{
  label: 'Shops',
  icon: <ShopIcon />,
  path: '/shop'
},
{
  label: 'Shop Categories',
  icon: <CategoryIcon />,
  path: '/shopCategory'
},
{
  label: 'Products',
  icon: <ShopIcon />,
  path: '/product'
},
{
  label: 'Product Categories',
  icon: <CategoryIcon />,
  path: '/productCategory'
},
{
  label: 'E-shop Users',
  icon: <PersonIcon />,
  path: '/eshopUser'
},
{
  label: 'Roles',
  icon: <RoleIcon />,
  path: '/role'
},
{
  label: 'Admins',
  icon: <AdminIcon />,
  path: '/admin'
},
{
  label: 'Carts',
  icon: <CartIcon />,
  path: '/cart'
},
{
  label: 'Cart Product',
  icon: <CartIcon />,
  path: '/cartProduct'
},
{
  label: 'Order',
  icon: <CartIcon />,
  path: '/orderCart'
},
{
  label: 'Payments',
  icon: <PaymentIcon />,
  path: '/payment'
},
{
  label: 'Payment Categories',
  icon: <PaymentIcon />,
  path: '/paymentCategory'
},
{
  label: 'Reviews',
  icon: <ReviewIcon />,
  path: '/review'
}
];

const menuItemsShopAdmin = [{
  label: 'Home',
  icon: <HomeIcon />,
  path: '/'
},
{
  label: 'Shops',
  icon: <ShopIcon />,
  path: '/shop'
},
{
  label: 'Products',
  icon: <ShopIcon />,
  path: '/product'
},
{
  label: 'Carts',
  icon: <CartIcon />,
  path: '/cart'
},
{
  label: 'Cart Product',
  icon: <CartIcon />,
  path: '/cartProduct'
},
{
  label: 'Order',
  icon: <CartIcon />,
  path: '/orderCart'
},
{
  label: 'Payments',
  icon: <PaymentIcon />,
  path: '/payment'
},
{
  label: 'Reviews',
  icon: <ReviewIcon />,
  path: '/review'
}
];

const menuItemsCustomer = [{
  label: 'Home',
  icon: <HomeIcon />,
  path: '/'
},
{
  label: 'Carts',
  icon: <CartIcon />,
  path: '/cart'
},
{
  label: 'Cart Product',
  icon: <CartIcon />,
  path: '/cartProduct'
},
{
  label: 'Order',
  icon: <CartIcon />,
  path: '/orderCart'
},
{
  label: 'Payments',
  icon: <PaymentIcon />,
  path: '/payment'
},
{
  label: 'Reviews',
  icon: <ReviewIcon />,
  path: '/review'
}
];

function LeftBar() {

  const [userName, setUserName] = useState('');
  const [role, setRole] = useState('');

  const handleUserChange = (newUserName) => {
    setUserName(newUserName);
  };

  const handleRoleChange = (newRole) => {
    setRole(newRole);
  };

  // Load user data on app startup
  useEffect(() => {
    setUserName(UserService.getUsername());
    setRole(UserService.getRole());
  }, []);

  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const handleMenuClick = () => {
    setOpen(!open);
  };

  const handleItemClick = (path) => {
    setOpen(false);
    navigate(path);
  };

  return (
    <>
      <IconButton
        edge="start"
        color="inherit"
        aria-label="menu"
        onClick={handleMenuClick}
      >
        <MenuIcon />
      </IconButton>
      <Drawer anchor="left" open={open} onClose={() => setOpen(false)}>
        {role === "SuperAdmin" && (
          <List>
            {menuItemsSuperAdmin.map((item) => (
              <ListItem button key={item.label} onClick={() => handleItemClick(item.path)}>
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItem>
            ))}
          </List>
        )}
        {role === "ShopAdmin" && (
          <List>
            {menuItemsShopAdmin.map((item) => (
              <ListItem button key={item.label} onClick={() => handleItemClick(item.path)}>
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItem>
            ))}
          </List>
        )}
        {role === "Customer" && (
          <List>
            {menuItemsCustomer.map((item) => (
              <ListItem button key={item.label} onClick={() => handleItemClick(item.path)}>
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItem>
            ))}
          </List>
        )}
        <Divider />
      </Drawer>
    </>
  );
}

export default LeftBar;
