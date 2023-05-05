import ShopDataDisplay from '../components/Shop/ShopTable'
import React from 'react';
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function Shop() {
    return (
        <div className="container">
            <h2>Our Shops!!!</h2>
            <span><br /><br /></span>
            <Toaster />
            <ShopDataDisplay />
            <div className="nav-bar-container-light">
                <BottomBar />
            </div>
        </div>
    );
}
export default Shop;