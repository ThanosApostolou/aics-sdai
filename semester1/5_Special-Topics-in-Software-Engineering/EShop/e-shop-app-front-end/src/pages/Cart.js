import CartDataDisplay from '../components/Cart/CartTable'
import React from 'react';
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function Cart() {
    return (
        <div className="container">
            <h2>Our Carts!!!</h2>
            <span><br /><br /></span>
            <Toaster />
            <CartDataDisplay />
            <div className="nav-bar-container-light">
                <BottomBar />
            </div>
        </div>
    );
}
export default Cart;