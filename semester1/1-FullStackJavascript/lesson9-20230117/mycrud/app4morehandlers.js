const fs =require("fs");
//import express
const express= require('express');

// create app variable -> assigned result of calling express function
// add a wide number of methods to app variable 
const app= express();

//before sending data we need to read it, we do this outside the route
//in the top level code that is excecuted once
//${__dirname}= where current script is located
var landmarks = fs.readFileSync(__dirname+'/dev-data/data/landmarks.json')

//convert json to javascript object
landmarks = JSON.parse(landmarks);

//(req,res)=>{} this callback funtion is called the route handler

app.get('/api/v1/landmarks',(req,res,next)=>{
    console.log("More than one callback can handle a route, dont' forget next!");
    next();
},(req,res)=>{
    res.status(200).json({
        status:"success",
        results: landmarks.length,
        data:{
            landmarks
        }
    });
})

var a=(req,res,next)=>{
    console.log("hi there");
    next();
}
var b=(req,res,next)=>{
    console.log("My friend");
    next();
};
var c=(req,res,)=>{
    res.send('Hello!')
};

app.get('/api/v1/array',[a,b,c]);

//start a server
app.listen(8080, () => {
    console.log('Yeah I run');
});