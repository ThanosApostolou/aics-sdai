package com.unipi.talepis.protal20222.designpatterns;

import java.util.ArrayList;
import java.util.List;

public class Parkadoros {
    private List<INotifyForCar> endiaferomenoi = new ArrayList<>();
    public void addListener(INotifyForCar iNotifyForCar){
        endiaferomenoi.add(iNotifyForCar);
    }
    public void parkingProblem(){
        System.out.println("A parking problem occurred./nLet me notify all");
        for (INotifyForCar iNotifyForCar : endiaferomenoi)
            iNotifyForCar.parkingProblemOccurred("Car parking problem");
    }
}
