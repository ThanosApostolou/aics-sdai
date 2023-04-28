(function () {
    const myobj = {
        name: 'test',
        myfunc: function () {
            console.log('myfunc: ' + this.name)
            function innerFunc() {
                console.log('innerFunc: ' + this.name)

            }

            const innerFunc2 = () => {
                console.log('innerFunc2: ' + this.name)

            }
            innerFunc.bind(this)();
            innerFunc2();
        }
    }

    myobj.myfunc();

    function MyType() {
        this.x = 2;
        this.y = 4;
    }

    const type1 = new MyType();
    const type2 = MyType();

    console.log('type1', type1)
    console.log('type2', type2)
    console.log('window', window)

    class MyClass {
        x = 3;
        y = 4;
    }

    MyClass.sz = 6;

    const myobj1 = new MyClass();
    // const myobj2 = MyClass();


    MyClass.prototype.z = 5;
    console.log('myobj1', myobj1.z)

    myobj1.z++;
    console.log('myobj1', myobj1.z)

    MyClass.prototype.z = 100;
    console.log('myobj1', myobj1.z)

    console.log('MyClass', MyClass.sz)
    // console.log('myobj2', myobj2)
    console.log('window', window)
})();