const express = require('express');
const { handleGetAllUsers } = require('./users-controller');

/***
 * @param {express.Router} parentRouter
 * @returns {void}
 */
exports.createUsersRouter = function (parentRouter) {
    const usersRouter = express.Router()
        .get('/fetch-all', handleGetAllUsers)

    parentRouter.use('/users', usersRouter)

}