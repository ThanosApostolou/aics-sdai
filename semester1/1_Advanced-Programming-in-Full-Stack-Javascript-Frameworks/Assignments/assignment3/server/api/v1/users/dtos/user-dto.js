exports.UserDto = class UserDto {

    /**
     * @param {object} obj
     * @param {string} obj._id
     * @param {string} obj.name
     * @param {string} obj.email
     * @param {number} obj.age
     * @param {string} obj.phone
     * @param {string} obj.address
     */
    constructor(obj) {
        this._id = obj?._id;
        this.name = obj?.name;
        this.email = obj?.email;
        this.age = obj?.age;
        this.phone = obj?.phone;
        this.address = obj?.address;
    }

    /**
     * @param {User} user
     * @returns {UserDto}
     */
    static fromUser(user) {
        if (!user) {
            return null;
        }
        return new UserDto(user);
    }

    /**
     * @param {object} body
     * @returns {UserDto}
     */
    static fromReqBody(body) {
        if (!body) {
            return null;
        }
        return new UserDto(body);
    }
}