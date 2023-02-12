const express = require('express');
const { handleGetAllUsers, handleCreateUser, handleUpdateUserById } = require('./users-controller');

/***
 * @param {express.Router} parentRouter
 * @returns {void}
 */
exports.createUsersRouter = function (parentRouter) {
    const usersRouter = express.Router()
        .get('/get-all-users', handleGetAllUsers)
        .post('/create-user', handleCreateUser)
        .put('/update-user-by-id', handleUpdateUserById)
        .post('/create-user', handleCreateUser)

    parentRouter.use('/users', usersRouter)

}