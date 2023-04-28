const { Vehicle } = require('../../../../../domain/vehicle/vehicle-entity');

exports.VehicleDto = class VehicleDto {

    /**
     * @param {object} obj
     * @param {string} obj._id
     * @param {string} obj.brand
     * @param {string} obj.type
     * @param {string} obj.model
     * @param {number} obj.year
     * @param {string} obj.engineCapacity
     * @param {string} obj.enginePower
     * @param {string} obj.photo
     */
    constructor(obj) {
        this._id = obj?._id;
        this.brand = obj?.brand;
        this.type = obj?.type;
        this.model = obj?.model;
        this.year = obj?.year;
        this.engineCapacity = obj?.engineCapacity;
        this.enginePower = obj?.enginePower;
        this.photo = obj?.photo;
    }

    /**
     * @param {Vehicle} vehicle
     * @returns {VehicleDto}
     */
    static fromVehicle(vehicle) {
        if (!vehicle) {
            return null;
        }
        return new VehicleDto(vehicle);
    }

    /**
     * @param {object} body
     * @returns {VehicleDto}
     */
    static fromReqBody(body) {
        if (!body) {
            return null;
        }
        return new VehicleDto(body);
    }
}
