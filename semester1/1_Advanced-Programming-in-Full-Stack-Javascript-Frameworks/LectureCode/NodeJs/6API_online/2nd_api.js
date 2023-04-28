const http = require('http');
const filesystem=require('fs');

//TOP LEVEL CODE
//use sync version, because it is easier to handle-> we put data in a var and we use it right away
// we do not care if data is blocking code execution because this top level code
//is going to be executed only once, at the beggining

const data=filesystem.readFileSync(`${__dirname}/countries.json`, "utf8");
console.log(data);

//create a server
//creatServer method will accept a callback function that will fire
//every time a new request hits our server 
const server= http.createServer((req, res) => {
    //we store the requested url in a variable
    const path_name= req.url;
    console.log(path_name);

    if(path_name==='/' || path_name==='/all'){
        res.writeHead('200',{'Content-Type': 'application/json'});
      //  res.writeHead('200',{'Content-Type': 'text/html'});

        res.end(data);
    }else if(path_name==='/api'){
            //lets send back the data we have parsed in the top level code
            //we need to let the browser know that we send back json
        
            res.writeHead('200',{'Content-Type': 'application/json'});
            res.end(data); //end the response
    }else{
        //inspect web page and check the console
        //we can also send headers, a piece of info regarding content
        res.writeHead('404',{'Content-Type': 'text/html'});
        
        res.end('<h2> 404 <br> Page not Found </h2>');
    }
    //res.end(); //end the response
  }); 
  

  //2nd part: listen to incoming requests from the client
  //port is a subaddress in a host
  //127.0.0.1 localhost
  server.listen(8080,'127.0.0.1',()=>{
      console.log('We are listening to requests on port 8080');
  });