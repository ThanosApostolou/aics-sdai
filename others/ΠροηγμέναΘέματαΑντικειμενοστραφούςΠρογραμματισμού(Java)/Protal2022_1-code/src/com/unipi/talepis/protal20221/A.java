package com.unipi.talepis.protal20221;

public class A implements MyInterface,OtherInterface{
    @Override
    public void doSomething() {
        System.out.println("I am speaking!");
    }

    @Override
    public void speak(String s) {
        System.out.println("Hi from "+s);
    }
}
