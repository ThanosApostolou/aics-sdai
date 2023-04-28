//relative path to our module within require
const cat=require('./module/mycats.js');
//if we just import a file it does not automatically gives as access to its vars


console.log(cat);
//console.log(cat.mycats, cat.ages);

//console.log(mycats);