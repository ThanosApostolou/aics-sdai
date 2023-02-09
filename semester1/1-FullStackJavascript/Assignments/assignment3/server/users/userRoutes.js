const express = require('express');
const userController = require(`${__dirname}/../controllers/userController`);

exports.createUserRouter = function () {

    const userRouter = express.Router();

    //User Routes
    userRouter.route('/')
        .get(userController.getAllUsers)
        .post(userController.addUser);


    userRouter.route('/:id')
        .get(userController.getUserById)
        .patch(userController.updateUserById)
        .delete(userController.deleteUserById);

    return userRouter;
}