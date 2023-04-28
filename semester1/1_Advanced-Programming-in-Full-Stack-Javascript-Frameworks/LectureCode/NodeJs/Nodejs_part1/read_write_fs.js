const filesystem=require('fs');
// readFileSync takes as arguments the filepath and the character encoding
var textread= filesystem.readFileSync(__dirname+'/txt/filetoread.txt','utf8');
console.log(textread);
//we create some text
var sometext= textread+'\n I will be your teacher for this course! :-) ';

//we write that text to a file that is created once the script is executed
//if we use this function to write in a file it will overwrite the content
filesystem.writeFileSync('./txt/output.txt',sometext);

//read file created or overwiten
var newtextread= filesystem.readFileSync(`${__dirname}/txt/output.txt`,'utf8');
console.log(newtextread); 