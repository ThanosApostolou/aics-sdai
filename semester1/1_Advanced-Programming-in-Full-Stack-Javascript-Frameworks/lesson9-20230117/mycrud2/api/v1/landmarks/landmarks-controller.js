const express = require('express');
const { GlobalState } = require('../../../infrastructure/global-state/global-state');

function buildAndSetLandmarksController(app) {
    const landmarksController = express.Router();
    landmarksController.route('/')
        .get(handleGetLandmarks)
    app.use('/api/v1/landmarks', landmarksController)
}

function handleGetLandmarks(req, res) {
    const landmarks = GlobalState.instance.landmarks;
    console.log('YEAH')

    res.status(200).json({
        status: "success",
        results: landmarks.length,
        data: {
            landmarks
        }
    });
}

module.exports = {
    buildAndSetLandmarksController,
    handleGetLandmarks
}