const http = require('http');

//.on method is how we actually create a listener,
//in this case for the "request" event.

const server = http.createServer()
//listen on the request event
server.on('request',(req, res) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/plain');
  console.log(req.url);
  console.log('Yeap, request was received');
  res.end('Hello World\n');
});

server.on('request',(req, res) => {
  console.log('hi again');});


server.listen(8080,'127.0.0.1',()=>{
    console.log('We are listening to requests on port 8080');
});