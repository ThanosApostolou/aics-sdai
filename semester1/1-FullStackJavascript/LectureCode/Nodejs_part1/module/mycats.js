const mycats =['rosa','javie','stilvi'];
const ages =[11,5,2];


console.log("I am in module:"+mycats);
//if we want to use aything from this file to another
//we need to manually export them

//module.exports="hello";

//module.exports=mycats;

//lets exploit more than one things
//we could export different properties of an object

module.exports={
    mycats:mycats,
    ages:ages
}


/*
module.exports={
    mycats,ages
}
*/
