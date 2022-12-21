const http = require('http');
const filesystem=require('fs');


const server= http.createServer((req, res) => {
    //we store the requested url in a variable
    const path_name= req.url;

    if(path_name==='/' || path_name==='/hi'){
        //response headers give the browser a little bit more information
        // regarding the response that is coming back
        res.writeHead('200',{'Content-Type': 'text/html'});
   
            //read and send html file as response
            filesystem.readFile('./views/index.html', (err, data)=>{
            if(err){
                res.end('<h2> 404 <br> Something wrong with the file</h2>');
                }else{
                res.end(data);
                }
            });
    }else{
        //inspect web page and check the console
        //we can also send headers, a piece of info about the response we are sending back
         //If the response from the HTTP server is supposed to be displayed as HTML,
         // we need to add HTTP header with the correct content type.

        res.writeHead('404',{'Content-Type': 'text/html'});
        
        res.end('<h2> 404 <br> Page not Found </h2>');
    }
    //res.end('<h2> 404 <br> Page not Found </h2>');
  }); 


  //2nd part: listen to incoming requests from the client
  //port is a subaddress in a host
  //127.0.0.1 localhost
  server.listen(8080,'127.0.0.1',()=>{
      console.log('We are listening to requests on port 8080');
  });