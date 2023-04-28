class EnvConfig {
    dbUser = '';
    dbPassword = '';
    dbHost = '';
    dbName = '';

    constructor(obj) {
        this.dbUser = obj.dbUser;
        this.dbPassword = obj.dbPassword;
        this.dbHost = obj.dbHost;
        this.dbName = obj.dbName;
    }

    static fromEnv() {
        return new EnvConfig({
            dbUser: process.env.DB_USER ? process.env.DB_USER : '',
            dbPassword: process.env.DB_PASSWORD ? process.env.DB_PASSWORD : '',
            dbHost: process.env.DB_HOST ? process.env.DB_HOST : '',
            dbName: process.env.DB_NAME ? process.env.DB_NAME : '',
        })
    }
}

module.exports = {
    EnvConfig
}