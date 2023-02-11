const express = require('express');
/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.handleGetAllUsers = (req, res, next) => {
    const users = [
        {
            name: 'test'
        }
    ]

    res.status(200).json({
        status: "success",
        results: users.length,
        data: users
    });
}
