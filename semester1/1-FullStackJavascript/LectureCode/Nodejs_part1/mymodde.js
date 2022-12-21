/*
destructuring assignment syntax
is a JavaScript expression that:
makes it possible to unpack values from arrays, 
or properties from objects, into distinct variables
*/
const {mycats}=require('./module/mycats.js');
//if we just import a file it does not automatically gives as access to its vars


console.log(mycats);

//console.log(mycats);