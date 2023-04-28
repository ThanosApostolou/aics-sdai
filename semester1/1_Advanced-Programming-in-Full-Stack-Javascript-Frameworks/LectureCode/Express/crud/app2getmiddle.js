const fs =require("fs");
const express= require('express');
const app= express();

var landmarks = fs.readFileSync(__dirname+'/dev-data/data/landmarks.json')
landmarks = JSON.parse(landmarks);

//handle get request app.get('/api/v1/landmarks'..
//(req,res)=>{} this callback funtion is called the route handler
app.use((req,res,next)=>{
    console.log("hi my friend, I am your middleware!")
    next();
});
app.use(secmiddle);
//app.use('/api/v1/',secmiddle);

function secmiddle(req,res,next){
    console.log("I live in a function")
    next();
}

app.get('/api/v1/landmarks',(req,res)=>{
    res.status(200).json({
        status:"success",
        results: landmarks.length,
        data:{
            landmarks
        }
    });
})

app.use(secmiddle);



//start a server
app.listen(8080, () => {
    console.log('Yeah I run');
});