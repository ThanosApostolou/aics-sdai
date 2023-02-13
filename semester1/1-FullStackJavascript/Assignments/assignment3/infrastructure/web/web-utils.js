const express = require('express');


exports.WebUtils = class {

    /**
     * @param {express.Response} res
     * @param {any} error
     * @return {express.Response<any, Record<string, any>>}
     */
    static sendInteralServerErrorResponse(res, error) {
        console.error(error);
        return res.status(500)
            .json({
                status: "error",
                message: 'unexpected error'
            });
    }

    /**
     * @param {express.Response} res
     * @param {object} result
     * @return {express.Response<any, Record<string, any>>}
     */
    static sendResponseByStatus(res, result) {
        if (result?.status === 'error') {
            console.error(result);
            return res.status(422)
                .json(result);
        } else {
            return res.status(200)
                .json(result);
        }
    }

    /**
     * @param {express.Request} req
     * @param {express.Response} res
     * @param {string} contentType
     * @return {boolean}
     */
    static isContentTypeInvalid(req, res, contentType) {
        const reqContentType = req.get('Content-Type');

        if (reqContentType && reqContentType !== contentType) {
            res.status(415)
                .json({
                    status: "error",
                    message: `expected Content-Type ${contentType} but got ${reqContentType}`
                });
            return true;
        }
        return false;
    }
}