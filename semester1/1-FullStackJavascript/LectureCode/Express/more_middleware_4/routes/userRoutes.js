
//import express
const express= require('express');
//import landmarkController module
const userController=require(`${__dirname}/../controllers/userController`);

const routes = express.Router();

//User Routes
routes.route('/')
.get(userController.getAllUsers)
.post(userController.addUser)


routes.route('/:id')
.get(userController.getUserById)
.patch(userController.updateUserById)
.delete(userController.deleteUserById)

//lets export our module!
module.exports= routes;