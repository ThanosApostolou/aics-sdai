const filesystem=require('fs');
// readFileSync takes as arguments the filepath and the character encoding

filesystem.readFile('./txt/not_embedded.txt', 'utf8', (err, data)=>{
    if(err) return console.log("oups!!!")
    //not_embedded.txt contains the word embedded, that is the name of the second file
    //${data} = + data +
    filesystem.readFile(`./txt/${data}.txt`, 'utf8', (err1, data1)=>{
      
        // Display the file content
        console.log(data1);
        console.log(err1);
    });
    
});

console.log('Reading file....');