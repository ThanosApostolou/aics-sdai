const { Vehicle } = require('../../../../domain/vehicle/vehicle-entity');
const { VehicleDto } = require('./dtos/vehicle-dto');
const { VehicleValidator } = require('../../../../domain/vehicle/vehicle-validator');

/**
 * @returns {Promise<object>}
 */
exports.doGetAllVehicles = async function () {
    const vehicles = await Vehicle.find();
    const vehiclesDtos = vehicles.map(vehicle => VehicleDto.fromVehicle(vehicle));
    return {
        status: "success",
        data: vehiclesDtos,
        dataLength: vehiclesDtos.length
    }
}

/**
 * @param {string} id
 * @returns {Promise<object>}
 */
exports.doGetVehicleById = async function (id) {
    if (!id) {
        return {
            status: "error",
            errors: `id was empty`
        };
    }
    if (id && typeof id !== 'string') {
        return {
            status: "error",
            errors: `id should be string`
        };
    }
    let vehicle;
    try {
        vehicle = await Vehicle.findById(id);

    } catch (e) {
        return {
            status: "error",
            errors: `could not find vehicle with id ${id}`
        };
    }
    if (!vehicle) {
        return {
            status: "error",
            errors: `could not find vehicle with id ${id}`
        };
    }
    const vehiclesDto = VehicleDto.fromVehicle(vehicle);
    return {
        status: "success",
        data: vehiclesDto
    }
}

/**
 * @param {VehicleDto} vehicleDto
 * @returns {Promise<object>}
 */
exports.doCreateVehicle = async function (vehicleDto) {
    const errors = [];

    const newVehicle = new Vehicle();
    newVehicle.brand = vehicleDto.brand;
    newVehicle.type = vehicleDto.type;
    newVehicle.model = vehicleDto.model;
    newVehicle.year = vehicleDto.year;
    newVehicle.engineCapacity = vehicleDto.engineCapacity;
    newVehicle.enginePower = vehicleDto.enginePower;
    newVehicle.photo = vehicleDto.photo;

    const syntaxErrors = VehicleValidator.validateSyntaxMandatory(newVehicle);
    if (syntaxErrors.length) {
        return {
            status: "error",
            errors: syntaxErrors
        };
    }

    const savedVehicle = await newVehicle.save()
    return {
        status: "success",
        errors: errors,
        savedVehicle: VehicleDto.fromVehicle(savedVehicle)
    };
}

/**
 * @param {string} id
 * @param {VehicleDto} vehicleDto
 * @returns {Promise<object>}
 */
exports.doUpdateVehicleById = async function (id, vehicleDto) {
    const errors = [];

    if (id == null) {
        return {
            status: "error",
            errors: "id was null"
        };
    }
    if (typeof id !== 'string') {
        return {
            status: "error",
            errors: "id was not string"
        };
    }

    try {
        const vehicle = await Vehicle.findById(id);
        if (!vehicle) {
            return {
                status: "error",
                errors: `could not find vehicle with id ${id}`
            };
        }

        vehicle.brand = vehicleDto.brand;
        vehicle.type = vehicleDto.type;
        vehicle.model = vehicleDto.model;
        vehicle.year = vehicleDto.year;
        vehicle.engineCapacity = vehicleDto.engineCapacity;
        vehicle.enginePower = vehicleDto.enginePower;
        vehicle.photo = vehicleDto.photo;

        const syntaxErrors = VehicleValidator.validateSyntaxMandatory(vehicle);
        if (syntaxErrors.length) {
            return {
                status: "error",
                errors: syntaxErrors
            };
        }

        const savedVehicle = await vehicle.save()
        return {
            status: "success",
            errors: errors,
            savedVehicle: VehicleDto.fromVehicle(savedVehicle)
        };
    } catch (e) {
        return {
            status: "error",
            errors: `could not find vehicle with id ${id}`
        };
    }


}

/**
 * @param {string} id
 * @returns {Promise<object>}
 */
exports.doDeleteVehicle = async function (id) {
    const errors = [];

    if (id == null) {
        return {
            status: "error",
            errors: "email was null"
        };
    }
    if (typeof id !== 'string') {
        return {
            status: "error",
            errors: "email was not string"
        };
    }

    let vehicle;
    try {
        vehicle = await Vehicle.findById(id);
        if (!vehicle) {
            return {
                status: "error",
                errors: `could not find vehicle with id ${id}`
            };
        }
    } catch (e) {
        return {
            status: "error",
            errors: `could not find vehicle with id ${id}`
        };

    }

    await vehicle.delete();

    return {
        status: "success",
    };


}