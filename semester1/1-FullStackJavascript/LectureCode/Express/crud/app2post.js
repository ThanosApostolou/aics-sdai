const fs =require("fs");
const express= require('express');
const app= express();

// middleware express.json() :is called middleware because 
//it is in the middle of the request and the response
//app.use('/api/v1/landmarks',express.json());
app.use(express.json());

var landmarks = fs.readFileSync(__dirname+'/dev-data/data/landmarks.json')
landmarks = JSON.parse(landmarks);

//lets add a new landmark
app.post('/api/v1/landmarks', (req,res)=>{
     console.log(req.body);
     res.status(200).json({
        status:"success"
    });
})

app.post('/api/v1/landmarksno', (req,res)=>{
    console.log(req.body);
    res.status(200).json({
        status:"success_sec"
    });
   
})

//start a server
app.listen(8080, () => {
    console.log('Yeah I run');
});