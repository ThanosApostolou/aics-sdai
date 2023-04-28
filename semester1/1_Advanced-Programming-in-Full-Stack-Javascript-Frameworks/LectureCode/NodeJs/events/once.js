
const EventEmitter = require('events');
const myEmitter = new EventEmitter();

myEmitter.once("done",()=>{
    console.log("I will run only once!");
});

myEmitter.emit('done');                
myEmitter.emit('done');             
console.log("finish")