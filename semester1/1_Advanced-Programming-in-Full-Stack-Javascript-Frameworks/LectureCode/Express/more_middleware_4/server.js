const dotenv = require('dotenv');
//specify path of configuration file, 
//'process'(nodejs app running) stores environment variables
//now using process.env.
// we can access our environment variables from everywhere
dotenv.config({ path: './config.env' });

//console.log(process.env);

const app = require('./app');

const port = process.env.PORT;
//START SERVER
app.listen(port, () => {
    console.log('Yeah I run');
});