package com.unipi.talepis.protal20222.designpatterns;

public class Professor implements INotifyForCar{
    @Override
    public void parkingProblemOccurred(String problem) {
        System.out.println("I am not leaving my lesson to fix my car problem:"+problem);
    }
}
