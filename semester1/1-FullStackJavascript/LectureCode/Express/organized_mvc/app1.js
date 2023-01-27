//--------------MODULES---------------------------
const fs = require('fs');
//import express
const express= require('express');
//import morgan
const morgan = require('morgan');

//--------------------------OTHER TOP LEVEL CODE-------------------------
// create app variable -> assigned result of calling express function
// add a wide number of methods to app variable 
const app= express();

//before sending data we need to read it, we do this outside the route
//in the top level code that is excecuted once
//${__dirname}= where current script is located
var landmarks = fs.readFileSync(`${__dirname}/dev-data/data/landmarks.json`)

//convert json to javascript object
landmarks = JSON.parse(landmarks);


//---------------------MIDDLEWARES---------------

//creating middleware express.json() :it is called middleware because 
//it is in the middle of the request and the response
app.use(express.json());

//3rd party middleware morgan
//app.use(morgan('dev'));

//my middleware
app.use((req,res,next)=>{
    console.log("hi my friend, I am your middleware!")
    next();
});

//---------------------ROUTEHANDLERS OR CONTROLERS----------------

//landmarks routehandler
const getAllLandmarks = (req,res)=>{

    res.status(200).json({
        status:"success",
        results: landmarks.length,
        data:{
            landmarks
        }
    });
}

const getLandmarkById = (req,res)=>{

    //req.params stores all parameters that we define in the url
    console.log(req.params)

    //remember in js when we multiply a string that looks like a number
    //with a number -> js converts string to number
    var id = req.params.id*1;
    var landmark = landmarks.find(element => element.id === id)
    if(!landmark){
       //if we do not put return here we will get error 
      //Error : Cannot set headers after they are sent to the client
      // because the response will be sent but code will continue running
      //so it will send another response also that is not allowed
       return res.status(404).json({
            status:"fail",
            message: "Not Found"
        });
    }
    res.status(200).json({
        status:"success",
        data: {
            landmark
        }
    });

}

const addLandmark= (req,res)=>{
    console.log(req.body);

    //lets store the data!
    landmarks.push(req.body);

    //here we are NOT going to use the syncronous 
    //as we are in the callback function

    //Convert landmarks from an object to a string with JSON.stringify 
    fs.writeFile(`${__dirname}/dev-data/data/landmarks.json`,JSON.stringify(landmarks),err=>{
       res.status(201).json({
           status:"success",
           data:{
               landmarks
           }
       });
    })
}

const updateLandmarkById = (req,res)=>{

    res.status(200).json({
        status:"success",
        data: {
            landmark:"Updated!"
        }
    });

}
const deleteLandmarkById = (req,res)=>{
    //204 status usually stands for no content
    //this is what we usually use with delete
    res.status(204).json({
        status:"success",
        data: null
    });

}

//user controllers
const getAllUsers = (req,res)=>{

    res.status(500).json({
        status:"error",
        message:"not yet defined"
    });
}

const getUserById = (req,res)=>{

    res.status(500).json({
        status:"error",
        message:"not yet defined"
    });

}

const addUser= (req,res)=>{
    res.status(500).json({
        status:"error",
        message:"not yet defined"
    });
}

const updateUserById = (req,res)=>{
    res.status(500).json({
        status:"error",
        message:"not yet defined"
    });

}
const deleteUserById = (req,res)=>{
    res.status(500).json({
        status:"error",
        message:"not yet defined"
    });

}


//----------------------------ROUTES------------------------

//lets create a new router-> it is a middleware
const landmarkRouter = express.Router();
const userRouter = express.Router();


//landmark Routes
landmarkRouter.route('/')
.get(getAllLandmarks)
.post(addLandmark)


landmarkRouter.route('/:id')
.get(getLandmarkById)
.patch(updateLandmarkById)
.delete(deleteLandmarkById)

//User Routes
userRouter.route('/')
.get(getAllUsers)
.post(addUser)


userRouter.route('/:id')
.get(getUserById)
.patch(updateUserById)
.delete(deleteUserById)

//here we specify the route(URL) for which we wish to use our middleware(landmarkRouter)
//We said that landmarksrouter is a middleware
//with use we define the path we want to use the middleware for
app.use('/api/v1/landmarks',landmarkRouter);
app.use('/api/v1/users',userRouter);

//START SERVER
app.listen(8080, () => {
    console.log('Yeah I run');
});