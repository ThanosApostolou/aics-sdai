package com.unipi.talepis.protal20221;

public class Demo {
    public static void main(String[] args) {
        SomeObject so1 = new SomeObject();
        giveMeFunctionality(so1);
        SomeObjectChild soc1 = new SomeObjectChild();
        giveMeFunctionality(soc1);
        giveMeFunctionality(new SomeObject(){
            @Override
            void sayHello() {
                System.out.println("Hiiiiii!!!!");
            }
        });
        System.out.println("#############################");
        A a1 = new A();
        giveMeFunctionalityV2(new A());
        giveMeFunctionalityV2(a1);
        giveMeFunctionalityV3(a1);
        giveMeFunctionalityV3(new OtherInterface() {
            @Override
            public void speak(String s) {
                System.out.println("My new code");
            }
        });
        giveMeFunctionalityV3(s -> System.out.println("My new code"));
    }
    static void giveMeFunctionality(SomeObject so){
        so.sayHello();
    }
    static void giveMeFunctionalityV2(MyInterface m){
        m.doSomething();
    }
    static void giveMeFunctionalityV3(OtherInterface o){
        o.speak("Unipi");
    }
}
