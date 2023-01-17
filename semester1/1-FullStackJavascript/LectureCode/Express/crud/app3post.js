const fs =require("fs");
//import express
const express= require('express');


// create app variable -> assigned result of calling express function
// add a wide number of methods to app variable 
const app= express();

//creating middleware express.json() :it is called middleware because 
//it is in the middle of the request and the response
app.use(express.json());

//before sending data we need to read it, we do this outside the route
//in the top level code that is excecuted once
//${__dirname}= where current script is located
var landmarks = fs.readFileSync(__dirname+'/dev-data/data/landmarks.json')

//convert json to javascript object
landmarks = JSON.parse(landmarks);

//handle get request app.get('/api/v1/landmarks'..
//I am specifing the url with version->good practice in some cases
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

//lets add a new landmark
//the url here shall remain the same, 
//the only thing that changes is the http method
app.post('/api/v1/landmarks', (req,res)=>{
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
})


//start a server
app.listen(8080, () => {
    console.log('Yeah I run');
});