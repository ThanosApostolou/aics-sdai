//import express
const express= require('express');
//import landmarkController module
const landmarkController=require(`${__dirname}/../controllers/landmarkController`);
//lets create a new router-> it is actually a middleware
const routes = express.Router();

routes.param(['id','x'], landmarkController.checkID);

//landmark Routes
routes.route('/')
.get(landmarkController.getAllLandmarks)
.post(landmarkController.checkBodyId,landmarkController.addLandmark)

routes.route('/:id/:x')
.get(landmarkController.getLandmarkById)
.patch(landmarkController.updateLandmarkById)
.delete(landmarkController.deleteLandmarkById)

//lets export our module!
module.exports= routes;
