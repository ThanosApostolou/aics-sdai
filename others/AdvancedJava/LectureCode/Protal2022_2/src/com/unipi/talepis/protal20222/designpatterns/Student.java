package com.unipi.talepis.protal20222.designpatterns;

public class Student implements INotifyForCar{
    @Override
    public void parkingProblemOccurred(String problem) {
        System.out.println("Lets go and see what's going on:"+problem);
    }
}
