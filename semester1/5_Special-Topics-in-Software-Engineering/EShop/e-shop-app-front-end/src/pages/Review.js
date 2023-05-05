import ReviewDataDisplay from '../components/Review/ReviewTable'
import React from 'react';
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function Review() {
    return (
        <div className="container">
            <h2>Our Reviews!!!</h2>
            <span><br /><br /></span>
            <Toaster />
            <ReviewDataDisplay />
            <div className="nav-bar-container-light">
                <BottomBar />
            </div>
        </div>
    );
}
export default Review;