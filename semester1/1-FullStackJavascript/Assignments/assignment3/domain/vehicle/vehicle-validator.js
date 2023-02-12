const { Vehicle } = require('./vehicle-entity');

exports.VehicleValidator = class VehicleValidator {

    /**
     * @param {Vehicle} vehicle
     * @returns {[string]}
     */
    static validateSyntaxMandatory(vehicle) {
        const errors = [];

        if (!vehicle) {
            errors.push('vehicle was null');
        }

        if (!vehicle.brand) {
            errors.push('vehicle brand was empty');
        }
        if (vehicle.brand != null && typeof vehicle.brand !== 'string') {
            errors.push('vehicle brand should be string');
        }

        if (!vehicle.type) {
            errors.push('vehicle type was empty');
        }
        if (vehicle.type != null && typeof vehicle.type !== 'string') {
            errors.push('vehicle type should be string');
        }

        if (!vehicle.model) {
            errors.push('vehicle model was empty');
        }
        if (vehicle.model != null && typeof vehicle.model !== 'string') {
            errors.push('vehicle model should be string');
        }

        if (!vehicle.engineCapacity) {
            errors.push('vehicle engineCapacity was empty');
        }
        if (vehicle.engineCapacity != null && typeof vehicle.model !== 'string') {
            errors.push('vehicle engineCapacity should be string');
        }

        if (!vehicle.enginePower) {
            errors.push('vehicle enginePower was empty');
        }
        if (vehicle.enginePower != null && typeof vehicle.model !== 'string') {
            errors.push('vehicle enginePower should be string');
        }

        if (vehicle.photo != null && typeof vehicle.model !== 'string') {
            errors.push('vehicle photo should be string');
        }

        if (vehicle.year == null) {
            errors.push('vehicle year was empty');
        }
        if (vehicle.year != null && typeof vehicle.year !== 'number') {
            errors.push('vehicle year should be number');
        }
        if (vehicle.year != null && !Number.isInteger(vehicle.year)) {
            errors.push('vehicle year must be integer');
        }
        if (vehicle.age != null && (vehicle.age < 1960 || vehicle.age > 2050)) {
            errors.push('vehicle age must be between 1960 and 2050');
        }
        return errors;
    }
}