import OrderCartDataDisplay from '../components/OrderCart/OrderCartTable'
import React from 'react';
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function OrderCart() {
    return (
        <div className="container">
            <h2>Our Orders!!!</h2>
            <span><br /><br /></span>
            <Toaster />
            <OrderCartDataDisplay />
            <div className="nav-bar-container-light">
                <BottomBar />
            </div>
        </div>
    );
}
export default OrderCart;