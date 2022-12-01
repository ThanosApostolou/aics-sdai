package com.unipi.talepis.protal20222.designpatterns;

public class Demo {
    public static void main(String[] args) {
        Parkadoros parkadoros = new Parkadoros();
        Professor p1 = new Professor();
        Student s1 = new Student();
        parkadoros.addListener(p1);
        parkadoros.addListener(s1);
        System.out.println("Unipi usual day");
        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        parkadoros.parkingProblem();
    }
}
