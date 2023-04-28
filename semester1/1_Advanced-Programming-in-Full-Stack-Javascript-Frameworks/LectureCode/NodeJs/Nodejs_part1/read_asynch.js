const filesystem=require('fs');

// simplest way to read a file in Node.js is to use the fs.readFile() method,
// passing it the file path, encoding and a callback 
//function that will be called with the file data (and the error):
filesystem.readFile('./txt/filetoread.txt', 'utf8', (mpa, kalaphge)=>{
      
    // Display the file content
    console.log(kalaphge);
    console.log(mpa);
});

  
console.log('readFile called');