import AdminDataDisplay from '../components/Admin/AdminTable'
import React from 'react';
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function Admin() {
    return (
        <div className="container">
            <h2>Our Admins!!!</h2>
            <span><br /><br /></span>
            <Toaster />
            <AdminDataDisplay />
            <div className="nav-bar-container-light">
                <BottomBar />
            </div>
        </div>
    );
}
export default Admin;