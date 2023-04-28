const filesystem = require('fs');
const mymod = require('./mymod.js')
const http = require('http');
const events = require('events');

function main() {
    const textread = filesystem.readFileSync(__dirname + '/test.txt');
    console.log('textread contents:', textread.toString())

    console.log('mymod', mymod.mycats)

    filesystem.readFile(__dirname + '/test.txt', 'utf-8', (err, data) => {
        console.log('data', data.toString())
    })

    test(() => {
        console.log('a')
    })
    test2(() => {
        console.log('b')
    })
    console.log('end main')

    const httpServer = http.createServer((req, res) => {
        console.log('req', req)
        console.log('res', res)
        // res.write('hello')
        // res.end();
        const url = req.url;

        if (url === '/home') {

            res.writeHead(200, {
                'Content-Type': 'text/html'
            })
            res.write('<h1>HELLO</h1>')
        } else if(url === '/about') {
            res.writeHead(200, {
                'Content-Type': 'text/html'
            })
            filesystem.readFileSync()
            res.write('<h1>HELLO</h1>')
        } else {
            res.writeHead(404, {
                'Content-Type': 'text/html'
            })
            res.write('<h1>Not FOund</h1>')
        }

        res.end()
    });

    httpServer.listen(8080, 'localhost', () => {
        console.log('hi we are listenging')
    })
}

function test(myCallback) {
    setTimeout(() => {
        myCallback()
    })
}

function test2(myCallback) {
    const eventEmmiter = new events.EventEmitter();
    eventEmmiter.addListener('emit', () => {
        myCallback();
    })
    eventEmmiter.emit('emit');
    eventEmmiter.removeAllListeners();
}

main();