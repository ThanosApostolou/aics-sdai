const express = require('express');
const { WebUtils } = require('../../../../infrastructure/web/web-utils');
const { VehicleDto } = require('./dtos/vehicle-dto');
const { doGetAllVehicles, doCreateVehicle, doUpdateVehicleById, doUpdateVehicleByEmail, doGetVehicleById, doGetVehicleByEmail: doGetVehicleNameByEmail, doDeleteVehicle } = require('./vehicles-actions');

/**
 * @param {express.Request} req
 * @param {express.Response} res
 * @param {express.NextFunction | undefined} next
 */
exports.handleGetAllVehicles = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const result = await doGetAllVehicles();
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
exports.handleGetVehicleById = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const id = req.query.id;
        const result = await doGetVehicleById(id);
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
exports.handleCreateVehicle = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const vehicleDto = VehicleDto.fromReqBody(req.body);
        const result = await doCreateVehicle(vehicleDto);

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
exports.handleUpdateVehicleById = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const id = req.query.id;
        const vehicleDto = VehicleDto.fromReqBody(req.body);
        const result = await doUpdateVehicleById(id, vehicleDto);

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
exports.handleDeleteVehicle = async (req, res, next) => {
    try {
        if (WebUtils.isContentTypeInvalid(req, res, 'application/json')) {
            return;
        }
        const id = req.query.id;
        const result = await doDeleteVehicle(id);
        return WebUtils.sendResponseByStatus(res, result);
    } catch (e) {
        WebUtils.sendInteralServerErrorResponse(res, e);
    }
}