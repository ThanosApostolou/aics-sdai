import CartProductDataDisplay from '../components/CartProduct/CartProductTable'
import React from 'react';
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function Admin() {
    return (
        <div className="container">
            <h2>Our Carts with Products!!!</h2>
            <span><br /><br /></span>
            <Toaster />
            <CartProductDataDisplay />
            <div className="nav-bar-container-light">
                <BottomBar />
            </div>
        </div>
    );
}
export default Admin;