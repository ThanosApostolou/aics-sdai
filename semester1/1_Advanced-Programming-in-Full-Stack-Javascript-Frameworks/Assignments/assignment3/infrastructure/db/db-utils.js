const mongoose = require('mongoose');
const { App } = require('../app/app');
const { EnvConfig } = require('../core/env-config');

exports.DbUtils = class {

    /**
     * @returns {Promise<typeof mongoose>}
     */
    static async connect() {
        const envConfig = App.instance().envConfig;
        const connectionString = `mongodb+srv://${envConfig.dbUser}:${envConfig.dbPassword}@${envConfig.dbHost}/${envConfig.dbName}?retryWrites=true&w=majority`
        return await mongoose.connect(connectionString);
    }
}