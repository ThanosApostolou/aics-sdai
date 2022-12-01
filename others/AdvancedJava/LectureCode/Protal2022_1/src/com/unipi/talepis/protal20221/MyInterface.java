package com.unipi.talepis.protal20221;

public interface MyInterface {
    static String hello = "Hi";
    void doSomething();
    default int add2Numbers(int a , int b){
        return a+b;
    }
    static void helperStaticMethod(){
        System.out.println("Give some help");
    }
}
