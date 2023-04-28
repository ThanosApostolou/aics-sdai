const express = require('express');
const { createUsersRouter } = require('./users/create-users-router');
const { createVehiclesRouter } = require('./vehicles/create-vehicles-router');

/**
 * @param {express.Router} parentRouter
 * @returns {void}
 */
exports.createV1Router = function (parentRouter) {
    const v1Router = express.Router();

    createUsersRouter(v1Router);
    createVehiclesRouter(v1Router);

    parentRouter.use('/v1', v1Router)

}