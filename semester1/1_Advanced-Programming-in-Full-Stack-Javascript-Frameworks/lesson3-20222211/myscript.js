(function () {
    var name = "Aristea";
    var surname;
    surname = "Kontogianni";
    console.log(name, surname);

    console.log('parse', Number(''))

    document.getElementById('myp').innerHTML = "<pre>" + JSON.stringify({ a: '23' }, null, 4) + "</pre>" //['a', 'b', 'c']

    // console.log('sort', ['a', 'b', null, undefined].map())
    const myl = ['a', 'b', null, undefined];
    myl.map(el => {
        console.log('el', el)
        return el;
    })

    // const myl2 = [1, 2, 3, 4]
    const myl2 = {
        0: 'a',
        1: 'b'
    }
    // const sum = myl2.reduce((prev, curr) => prev + curr, 0)
    // const avg = sum / myl2.length;
    // console.log('avg', avg)

    for (const [index, val] of Object.entries(myl2)) {
        console.log('for', index, val)
    }

    function parseString(mystr) {
        if (str === null || str === undefined) {
            return null;
        }
        if (mystr !== 'string') {
            throw new Error('not a string')
        }


    }

})();