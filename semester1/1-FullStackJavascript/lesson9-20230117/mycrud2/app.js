const fs =require("fs");
//import express
const express= require('express');
const { buildAndSetLandmarksController } = require("./api/v1/landmarks/landmarks-controller");
const { GlobalState } = require("./infrastructure/global-state/global-state");


// create app variable -> assigned result of calling express function
// add a wide number of methods to app variable 
const app= express();

//creating middleware express.json() :it is called middleware because 
//it is in the middle of the request and the response
app.use(express.json());


app.use((req,res,next)=>{
     console.log("hi my friend, I am your middleware!")
     next();
});
//before sending data we need to read it, we do this outside the route
//in the top level code that is excecuted once
//${__dirname}= where current script is located
var landmarks = fs.readFileSync(`${__dirname}/dev-data/data/landmarks.json`)

//convert json to javascript object
landmarks = JSON.parse(landmarks);

GlobalState.initialize(landmarks);

//handle get requests app.get('/api/v1/landmarks'..
//I am specifing the url with version->good practice in some cases
//(req,res)=>{} this callback funtion is called the route handler
app.get('/api/v1/landmarks/:id', (req,res)=>{

     console.log(req.params);
    // in js when we multiply a string that looks like a number
    //with a number -> js converts string to number
    var id = req.params.id*1;
    console.log(typeof(id));
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
    


})

// app.get('/api/v1/landmarks', (req,res)=>{

//     res.status(200).json({
//         status:"success",
//         results: landmarks.length,
//         data:{
//             landmarks
//         }
//     });
// })

buildAndSetLandmarksController(app);

app.get('/api/v1/landmarks/:id/:x/:kati', (req,res)=>{

    
     //req.params stores all parameters that we define in the url
     console.log(req.params)
     res.send("ok")
 

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

//update one property with patch
//we shall see more in databases about that
app.patch('/api/v1/landmarks/:id', (req,res)=>{

    res.status(200).json({
        status:"success",
        data: {
            landmark:"Updated!"
        }
    });

})

//we shall see more in databases about that
app.delete('/api/v1/landmarks/:id', (req,res)=>{
    //204 status usually stands for no content
    //this is what we usually use with delete
    res.status(204).json({
        status:"success",
        data: null
    });

})
//start a server
app.listen(8080, () => {
    console.log('Yeah I run');
});