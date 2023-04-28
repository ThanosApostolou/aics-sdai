const express = require('express');
const { handleGetAllVehicles, handleCreateVehicle, handleUpdateVehicleById, handleUpdateVehicleByEmail, handleGetVehicleById, handleGetVehicleByEmail: handleGetVehicleNameByEmail, handleDeleteVehicle } = require('./vehicles-controller');

/***
 * @param {express.Router} parentRouter
 * @returns {void}
 */
exports.createVehiclesRouter = function (parentRouter) {
    const vehiclesRouter = express.Router()
        .get('/get-all-vehicles', handleGetAllVehicles)
        .get('/get-vehicle-by-id', handleGetVehicleById)
        .post('/create-vehicle', handleCreateVehicle)
        .put('/update-vehicle-by-id', handleUpdateVehicleById)
        .delete('/delete-vehicle', handleDeleteVehicle)

    parentRouter.use('/vehicles', vehiclesRouter)

}