const EventEmitter = require('events');

// create an instance of the imported class
//create an object of EventEmitter class from events module
const myEmitter = new EventEmitter();

//listen to an event, we get arguments from emitter
//this is a custom event
myEmitter.on('hi', (data,test)=>{
    console.log('First event: ' + data+" ..."+test);
 });
 myEmitter.on('hi', ()=>{
    console.log('I am the second listener');
 });

 // Raising hi event: object that emits an event
 //myEmitter.emit('hi', 'My first Node.js event has been triggered.');

 myEmitter.emit('hi', 'My first Node.js event has been triggered.',"kati");

