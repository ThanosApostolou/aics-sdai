// Importing events
const EventEmitter = require('events');
  
// Initializing event emitter instances 
var eventEmitter = new EventEmitter();
   
var fun1= (msg) => {
    console.log("Hello everyone I am " + msg);
};
   
var fun2 = (msg) => {
    console.log("I do not know what to console.log "+msg);
};
  
// Registering listeners for events: 
eventEmitter.on('myEvent', fun1);
eventEmitter.on('myEvent', fun2);
   
// Triggering myEvent
eventEmitter.emit('myEvent', "An event");

  // Removing listener fun1
eventEmitter.removeListener('myEvent', fun1);
  
  
// Triggering myEvent
eventEmitter.emit('myEvent', "Event occurred again !");

// Removing all the listeners to myEvent
eventEmitter.removeAllListeners('myEvent');

  
// Triggering myEvent
eventEmitter.emit('myEvent', "One more time!");