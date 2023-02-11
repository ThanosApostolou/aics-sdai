const { createServer } = require('./server/create-server');
const dotenv = require('dotenv');
const { initializeApp, App } = require('./infrastructure/app/app');
const { DbUtils } = require('./infrastructure/db/db-utils');

async function main() {
    dotenv.config({
        path: `${__dirname}/.env`
    })

    initializeApp();

    console.log('instance', App.instance());
    // mongodb+srv://<username>:<password>@cluster0.sj8dgrp.mongodb.net/?retryWrites=true&w=majority

    const connection = await DbUtils.connect();

    const app = createServer();

    //START SERVER
    app.listen(8080, () => {
        console.log('Yeah I run');
    });
}

main();
