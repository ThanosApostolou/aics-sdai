const express = require('express');
const { WebUtils } = require('../../../../infrastructure/web/web-utils');
const { UserDto } = require('./dtos/user-dto');
const { doGetAllUsers, doCreateUser, doUpdateUserById, doUpdateUserByEmail, doGetUserById, doGetUserNameByEmail, doDeleteUser } = require('./users-actions');

/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.handleGetAllUsers = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const result = await doGetAllUsers();
        return WebUtils.sendResponseByStatus(res, result);
    } catch (e) {
        WebUtils.sendInteralServerErrorResponse(res, e);
    }
}


/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.handleGetUserById = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const id = req.query.id;
        const result = await doGetUserById(id);
        return WebUtils.sendResponseByStatus(res, result);
    } catch (e) {
        WebUtils.sendInteralServerErrorResponse(res, e);
    }
}

/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.handleGetUserNameByEmail = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const email = req.query.email;
        const result = await doGetUserNameByEmail(email);
        return WebUtils.sendResponseByStatus(res, result);
    } catch (e) {
        WebUtils.sendInteralServerErrorResponse(res, e);
    }
}

/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.handleCreateUser = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const userDto = UserDto.fromReqBody(req.body);
        const result = await doCreateUser(userDto);

        return WebUtils.sendResponseByStatus(res, result)
    } catch (e) {
        WebUtils.sendInteralServerErrorResponse(res, e);
    }
}

/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.handleUpdateUserById = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const id = req.query.id;
        const userDto = UserDto.fromReqBody(req.body);
        const result = await doUpdateUserById(id, userDto);

        return WebUtils.sendResponseByStatus(res, result)
    } catch (e) {
        WebUtils.sendInteralServerErrorResponse(res, e);
    }
}


/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.handleUpdateUserByEmail = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const email = req.query.email;
        const userDto = UserDto.fromReqBody(req.body);
        const result = await doUpdateUserByEmail(email, userDto);

        return WebUtils.sendResponseByStatus(res, result)
    } catch (e) {
        WebUtils.sendInteralServerErrorResponse(res, e);
    }
}


/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.handleDeleteUser = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const id = req.query.id;
        const result = await doDeleteUser(id);
        return WebUtils.sendResponseByStatus(res, result);
    } catch (e) {
        WebUtils.sendInteralServerErrorResponse(res, e);
    }
}