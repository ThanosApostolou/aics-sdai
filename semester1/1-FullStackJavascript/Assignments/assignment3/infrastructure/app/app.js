const { EnvConfig } = require('../core/env-config');

class App {
    /**
     * @type {App}
     */
    static _instance = null;

    /**
     * @returns {App}
     */
    static instance() {
        if (this._instance) {
            return this._instance;
        }
        throw new Error('App not initialized');
    }

    envConfig

    constructor(obj) {
        this.envConfig = obj.envConfig;
    }

}

function initializeApp() {
    App._instance = new App({
        envConfig: EnvConfig.fromEnv()
    })
}

module.exports = {
    App,
    initializeApp
}