import './App.css'
import { Routes, Route } from "react-router-dom"
import Shop from "./pages/Shop"
import ShopCategory from "./pages/ShopCategory"
import EshopUser from "./pages/EshopUser"
import Role from "./pages/Role"
import Admin from "./pages/Admin"
import Review from "./pages/Review"
import Payment from "./pages/Payment"
import PaymentCategory from "./pages/PaymentCategory"
import Product from "./pages/Product"
import Cart from "./pages/Cart"
import CartProduct from "./pages/CartProduct"
import OrderCart from "./pages/OrderCart"
import ProductCategory from "./pages/ProductCategory"
import Home from "./pages/Home"
import About from "./pages/About"
import Location from "./pages/Location"
import LeftBar from "./commons/LeftBar"
import UserService from './services/UserService';
import { useState, useEffect } from 'react';

function App() {

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

  return (
    <div className="App">
      <LeftBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/home" element={<Home />} />
        <Route path="/location" element={<Location />} />
        <Route path="/about" element={<About />} />
        <Route path="/cart" element={<Cart />} />
        <Route path="/cartProduct" element={<CartProduct />} />
        <Route path="/orderCart" element={<OrderCart />} />
        <Route path="/review" element={<Review />} />
        <Route path="/payment" element={<Payment />} />
        {role === "SuperAdmin" || role === "ShopAdmin" ? (
          <>
            <Route path="/shop" element={<Shop />} />
            <Route path="/product" element={<Product />} />
          </>
        ) : null}
        {role === "SuperAdmin" ? (
          <>
            <Route path="/shopCategory" element={<ShopCategory />} />
            <Route path="/role" element={<Role />} />
            <Route path="/admin" element={<Admin />} />
            <Route path="/productCategory" element={<ProductCategory />} />
            <Route path="/eshopUser" element={<EshopUser />} />
            <Route path="/paymentCategory" element={<PaymentCategory />} />
          </>
        ) : null}
      </Routes>
    </div>
  );
}

export default App;
