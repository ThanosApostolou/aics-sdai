const express = require('express');
const { handleGetAllUsers, handleCreateUser, handleUpdateUserById, handleUpdateUserByEmail, handleGetUserById, handleGetUserNameByEmail, handleDeleteUser } = require('./users-controller');

/***
 * @param {express.Router} parentRouter
 * @returns {void}
 */
exports.createUsersRouter = function (parentRouter) {
    const usersRouter = express.Router()
        .get('/get-all-users', handleGetAllUsers)
        .get('/get-user-by-id', handleGetUserById)
        .get('/get-user-name-by-email', handleGetUserNameByEmail)
        .post('/create-user', handleCreateUser)
        .put('/update-user-by-id', handleUpdateUserById)
        .put('/update-user-by-email', handleUpdateUserByEmail)
        .delete('/delete-user', handleDeleteUser)

    parentRouter.use('/users', usersRouter)

}