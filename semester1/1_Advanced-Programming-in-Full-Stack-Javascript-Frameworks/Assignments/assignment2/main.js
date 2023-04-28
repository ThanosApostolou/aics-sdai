const http = require('http');
const filesystem = require('fs');
const url = require('url');

function main() {
    try {
        const programmingLJson = filesystem.readFileSync(`${__dirname}/messages/programmingL.json`, "utf8");
        const notFoundJson = filesystem.readFileSync(`${__dirname}/messages/not_found.json`, "utf8");

        const server = http.createServer((req, res) => {
            const urlWithParsedQuery = url.parse(req.url, true);
            const pathName = urlWithParsedQuery.pathname;
            const query = urlWithParsedQuery.query;

            if (pathName === '/' || pathName === '/overview') {
                res.writeHead('200', { 'Content-Type': 'application/json' });
                res.end(programmingLJson)
            } else if (pathName === '/program') {
                try {
                    if (query == null || query.id == null) {
                        res.writeHead('400', { 'Content-Type': 'application/json' });
                        const responseBody = {
                            error: "expected id query parameter"
                        };
                        res.end(JSON.stringify(responseBody));
                    } else {
                        const id = parseInt(query.id);
                        if (isNaN(id)) {
                            res.writeHead('400', { 'Content-Type': 'application/json' });
                            const responseBody = {
                                error: "id query parameter was not a number"
                            };
                            res.end(JSON.stringify(responseBody));
                        } else {
                            const programmingLObj = JSON.parse(programmingLJson);
                            const programl = programmingLObj.programmingL[id];
                            res.writeHead('200', { 'Content-Type': 'application/json' });
                            res.end(JSON.stringify(programl));
                        }
                    }
                } catch (e) {
                    console.error(e);
                    res.writeHead('500', { 'Content-Type': 'application/json' });
                    res.end()
                }
            } else {
                res.writeHead('404', { 'Content-Type': 'text/html' });
                res.end(notFoundJson);
            }
        });

        server.listen(8080, '127.0.0.1', () => {
            console.log('We are listening to requests on http://127.0.0.1:8080');
        });
    } catch (e) {
        if (e instanceof Error) {
            console.err(e.message);
        } else {
            console.error('something went wrong');
        }
    }
}

main();