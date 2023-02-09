const { createServer } = require('./server/create-server');

function main() {

    const app = createServer();

    //START SERVER
    app.listen(8080, () => {
        console.log('Yeah I run');
    });
}

main();
