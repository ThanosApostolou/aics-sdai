const express = require('express');
const morgan = require('morgan');
const { createApiRouter } = require('./api/create-api-router');

exports.createServer = function () {
    const expressInstance = express();

    //---------------------MIDDLEWARES----------------
    expressInstance.use(express.json());
    //serve static files
    expressInstance.use(express.static(`${__dirname}/public`));
    //3rd party middleware morgan :HTTP request logger middleware for node.js
    expressInstance.use(morgan('dev'));
    //my middleware
    expressInstance.use((req, res, next) => {
        console.log("hi my friend, I am your middleware!")
        next();
    });

    createApiRouter(expressInstance);

    return expressInstance;
}
