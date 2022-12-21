const filesystem=require('fs');
// readFileSync takes as arguments the filepath and the character encoding
//var textread= filesystem.readFileSync(__dirname+'/txt/filetoread.txt','utf8');
//var textread= filesystem.readFileSync('./txt/filetoread.txt','utf8');
var textread= filesystem.readFileSync('./first_app/txt/filetoread.txt','utf8');


console.log(textread);
