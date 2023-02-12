const { User } = require('../../../../domain/user/user-entity');
const { UserValidator } = require('../../../../domain/user/user-validator');
const { UserDto } = require('./dtos/user-dto');

/**
 * @returns {Promise<[string]>}
 */
exports.doGetAllUsers = async function () {
    const users = await User.find();
    const usersDtos = users.map(user => UserDto.fromUser(user));
    return {
        status: "success",
        data: usersDtos,
        dataLength: usersDtos.length
    }
}

/**
 * @param {UserDto} userDto
 * @returns {Promise<[string]>}
 */
exports.doCreateUser = async function (userDto) {
    const errors = [];

    const newUser = new User();
    newUser.name = userDto.name;
    newUser.email = userDto.email;
    newUser.age = userDto.age;
    newUser.phone = userDto.phone;
    newUser.address = userDto.address;

    const syntaxErrors = UserValidator.validateSyntaxMandatory(newUser);
    if (syntaxErrors.length) {
        return {
            status: "error",
            errors: syntaxErrors
        };
    }

    const br1Error = await UserValidator.validateBr1UniqueEmail(newUser);
    if (br1Error) {
        return {
            status: "error",
            errors: [br1Error]
        };
    }

    const savedUser = await newUser.save()
    return {
        status: "success",
        errors: errors,
        savedUser: UserDto.fromUser(savedUser)
    };
}

/**
 * @param {string} id
 * @param {UserDto} userDto
 * @returns {Promise<[string]>}
 */
exports.doUpdateUserById = async function (id, userDto) {
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
        const user = await User.findById(id);
        if (!user) {
            return {
                status: "error",
                errors: `could not find user with id ${id}`
            };
        }

        user.name = userDto.name;
        user.email = userDto.email;
        user.age = userDto.age;
        user.phone = userDto.phone;
        user.address = userDto.address;

        const syntaxErrors = UserValidator.validateSyntaxMandatory(user);
        if (syntaxErrors.length) {
            return {
                status: "error",
                errors: syntaxErrors
            };
        }

        console.log('user.id', user.id)
        const br1Error = await UserValidator.validateBr1UniqueEmail(user);
        if (br1Error) {
            return {
                status: "error",
                errors: [br1Error]
            };
        }

        const savedUser = await user.save()
        return {
            status: "success",
            errors: errors,
            savedUser: UserDto.fromUser(savedUser)
        };
    } catch (e) {
        return {
            status: "error",
            errors: `could not find user with id ${id}`
        };
    }


}


/**
 * @param {string} email
 * @param {UserDto} userDto
 * @returns {Promise<[string]>}
 */
exports.doUpdateUserByEmail = async function (email, userDto) {
    const errors = [];

    if (email == null) {
        return {
            status: "error",
            errors: "email was null"
        };
    }
    if (typeof email !== 'string') {
        return {
            status: "error",
            errors: "email was not string"
        };
    }


    try {
        const user = await User.findOne({
            email: email
        });
        if (!user) {
            return {
                status: "error",
                errors: `could not find user with email ${email}`
            };
        }

        user.name = userDto.name;
        user.email = userDto.email;
        user.age = userDto.age;
        user.phone = userDto.phone;
        user.address = userDto.address;

        const syntaxErrors = UserValidator.validateSyntaxMandatory(user);
        if (syntaxErrors.length) {
            return {
                status: "error",
                errors: syntaxErrors
            };
        }

        const br1Error = await UserValidator.validateBr1UniqueEmail(user);
        if (br1Error) {
            return {
                status: "error",
                errors: [br1Error]
            };
        }

        const savedUser = await user.save()
        return {
            status: "success",
            errors: errors,
            savedUser: UserDto.fromUser(savedUser)
        };
    } catch (e) {
        return {
            status: "error",
            errors: `could not find user with email ${email}`
        };

    }


}