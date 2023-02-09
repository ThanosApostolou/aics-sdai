//-----------------MODULES---------------------
//import express
const express = require('express');
//import morgan
const morgan = require('morgan');

exports.createServer = function () {
    const landmarkRouter = require('./routes/landmarkRoutes')
    const userRouter = require('./routes/userRoutes')
    //---------------------------------------------
    // create app variable -> assigned result of calling express function
    const app = express();

    //---------------------MIDDLEWARES----------------
    app.use(express.json());
    //serve static files
    app.use(express.static(`${__dirname}/public`));
    //3rd party middleware morgan :HTTP request logger middleware for node.js
    app.use(morgan('dev'));

    //my middleware
    app.use((req, res, next) => {
        console.log("hi my friend, I am your middleware!")
        next();
    });

    //here we specify the route(URL) for which we wish to use our middleware(landmarkRouter)
    app.use('/api/v1/landmarks', landmarkRouter);
    app.use('/api/v1/users', userRouter);

    return app;
}
