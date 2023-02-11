const express = require('express');
const { createV1Router } = require('./v1/create-v1-router');

/**
 * @param {express.Express} expressInstance
 * @returns {void}
 */
exports.createApiRouter = function (expressInstance) {
    const apiRouter = express.Router();

    createV1Router(apiRouter);

    expressInstance.use('/api', apiRouter)

}