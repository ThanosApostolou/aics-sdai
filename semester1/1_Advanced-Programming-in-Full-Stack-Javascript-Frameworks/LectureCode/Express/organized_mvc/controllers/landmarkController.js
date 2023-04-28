//we need fs here because we open the json file
const fs = require('fs');

//${__dirname}= where current script is located
var landmarks = fs.readFileSync(`${__dirname}/../dev-data/data/landmarks.json`)

//convert json to javascript object
landmarks = JSON.parse(landmarks);


exports.checkID = (req, res, next, val) => {
    console.log(`Id is: ${val}`);
    console.log(`Id is: ${val1}`);

  
    var id = req.params.id*1;
    var landmark = landmarks.find(element => element.id === id)
    if(!landmark){
      return res.status(404).json({
        status: 'fail',
        message: 'Invalid ID! Sorry!'
      });
    }
    next();
  };

  exports.checkBodyId = (req, res, next) => {
    var id = req.body.id*1;
    var landmark = landmarks.find(element => element.id == id)
    console.log(landmark);
    if (landmark) {
      return res.status(400).json({
        status: 'fail',
        message: 'This id already Exists'
      });
    }
    next();
  };
  

//-------------------------landmarks routehandler/ controllers--------------------
//we want to export ALL functions from this module
//we are going to put all these functions to the exports object
exports.getAllLandmarks = (req,res)=>{

    res.status(200).json({
        status:"success",
        results: landmarks.length,
        data:{
            landmarks
        }
    });
}

exports.getLandmarkById = (req,res)=>{

    //req.params stores all parameters that we define in the url
    console.log(req.params)

    //remember in js when we multiply a string that looks like a number
    //with a number -> js converts string to number
    var id = req.params.id*1;
    var landmark = landmarks.find(element => element.id === id)
   
    res.status(200).json({
        status:"success",
        data: {
            landmark
        }
    });

}



exports.addLandmark= (req,res)=>{
    console.log(req.body);
    console.log("HI");

    //lets store the data!
    landmarks.push(req.body);

    //here we are NOT going to use the syncronous 
    //as we are in the callback function

    
    //Convert landmarks from an object to a string with JSON.stringify 
    fs.writeFile(`${__dirname}/../dev-data/data/landmarks.json`,JSON.stringify(landmarks),err=>{
        res.status(201).json({
            status:"success",
            data:{
                landmarks
            }
        });
     })
}

exports.updateLandmarkById = (req,res)=>{

    res.status(200).json({
        status:"success",
        data: {
            landmark:"Updated!"
        }
    });

}
exports.deleteLandmarkById = (req,res)=>{
    //204 status usually stands for no content
    //this is what we usually use with delete
    res.status(204).json({
        status:"success",
        data: null
    });

}



