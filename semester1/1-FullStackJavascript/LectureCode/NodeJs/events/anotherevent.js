const filesystem=require('fs');
const EventEmitter = require('events');
// create an instance of the imported class
//create an object of EventEmitter class from events module
const myEmitter = new EventEmitter();

myEmitter.on('update', ()=>{
    console.log('File succesfully updated');
 });

filesystem.readFile('./txt/not_embedded.txt', 'utf8', (err, data)=>{
    if(err) return console.log("oups!!!")

        filesystem.writeFile(`./txt/newfile.txt`,data, (err) => {
            if (err) return console.log ('Could not write file 😢');
            myEmitter.emit('update');
          });

 
});
  
console.log('Reading file....');