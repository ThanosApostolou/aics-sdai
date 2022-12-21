const http = require('http');
//create a server
//creatServer method will accept a callback function that will fire
//every time a new request hits our server 

//1st part: create a server object
const server= http.createServer((req, res) => {
   
  //console.log(req); 
  console.log(req.url, req.method);
  res.write('hello');
  res.end();
  }); //the server object listens on port 8080

  //2nd part: listen to incoming requests from the client
  //port is a subaddress in a host 
  //host is : 127.0.0.1 localhost
  //callback function is optional
  server.listen(8080,'localhost',()=>{
      console.log('hI! We are listening to requests! ');
  });