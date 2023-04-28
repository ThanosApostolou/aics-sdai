const fs =require("fs");
//import express
const express= require('express');

/* create app variable -> 
assigned result of calling express function
add a wide number of methods to app variable 
*/
const app= express();

//before sending data we need to read it

var landmarks = fs.readFileSync(__dirname+'/dev-data/data/landmarks.json')

//convert json to javascript object
landmarks = JSON.parse(landmarks);

//handle get request app.get('/api/v1/landmarks'..
//I am specifing the path with version->good practice in some cases
//(req,res)=>{} this callback funtion is called the route handler

app.get('/api/v1/landmarks', (req,res)=>{

    res.status(200).json({
        status:"success",
        results: landmarks.length,
        data:{
            landmarks
        }
    });
})

//start a server
app.listen(8080, () => {
    console.log('Yeah I run');
});