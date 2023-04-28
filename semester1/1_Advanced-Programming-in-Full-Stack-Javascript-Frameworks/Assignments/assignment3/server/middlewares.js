const express = require('express');
const { UserDto } = require('./api/v1/users/dtos/user-dto');

/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.userAgeMiddleware = function (req, res, next) {
    const usersPrefix = '/api/v1/users';
    const validPaths = [
        `${usersPrefix}/create-user`,
        `${usersPrefix}/update-user-by-id`,
        `${usersPrefix}/update-user-by-email`
    ]

    // if path is in the valid paths of this middleware
    if (validPaths.includes(req.path)) {
        const userDto = UserDto.fromReqBody(req.body)
        if (userDto?.age != null
            && typeof userDto.age === 'number'
            && userDto.age < 18
        ) {
            return res.status(422)
                .json({
                    status: "error",
                    error: "age must be greater or equal to 18"
                });
        } else {
            next();
        }
    } else {
        next();
    }
}