const { User } = require('./user-entity');

exports.UserValidator = class UserValidator {

    /**
     * @param {User} user
     * @returns {[string]}
     */
    static validateSyntaxMandatory(user) {
        const errors = [];

        if (!user) {
            errors.push('user was null');
        }

        if (!user.name) {
            errors.push('user name was empty');
        }
        if (user.name != null && typeof user.name !== 'string') {
            errors.push('user name should be string');
        }

        if (!user.email) {
            errors.push('user email was empty');
        }
        if (user.email != null && typeof user.email !== 'string') {
            errors.push('user email should be string');
        }

        if (user.age == null) {
            errors.push('user age was empty');
        }
        if (user.age != null && typeof user.age !== 'number') {
            errors.push('user email should be number');
        }
        if (user.age != null && !Number.isInteger(user.age)) {
            errors.push('user age must be integer');
        }
        if (user.age != null && (user.age < 1 || user.age > 150)) {
            errors.push('user age must be between 1 and 150');
        }

        if (user.phone != null && typeof user.phone !== 'string') {
            errors.push('user phone should be string');
        }

        if (user.adress != null && typeof user.adress !== 'string') {
            errors.push('user adress should be string');
        }

        return errors;
    }

    /**
     * Business rule 1. Validates uniqune email.
     * @param {User} user
     * @returns {Prmoise<string>}
     */
    static async validateBr1UniqueEmail(user) {
        if (user?.email) {
            const emailExists = await User.exists({
                _id: {
                    $ne: user._id,
                },
                email: user.email
            })
            if (emailExists) {
                return `email ${user.email} is already in use`
            } else {

                return null;
            }
        }
        return null;
    }
}