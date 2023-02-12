const express = require('express');
const { WebUtils } = require('../../../../infrastructure/web/web-utils');
const { UserDto } = require('./dtos/user-dto');
const { doGetAllUsers, doCreateUser, doUpdateUserById } = require('./users-actions');

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