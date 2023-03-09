package com.unipi.talepis.protal20223;

import java.util.Random;

public class Demo3b {
    private static String message;
    public static void main(String[] args) {
        Object mylock = new Object();
        Random r = new Random();
        int n = r.nextInt(5);
        Thread thread1 = new Thread(()->{
            double counter = 0;
            //synchronized (mylock){
                while (message==null){
                    counter++;
                    /*try {
                        //mylock.wait();
                        System.out.println(Thread.currentThread().getName()+" waiting");
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }*/
                    System.out.println(Thread.currentThread().getName()+" waiting "+counter);
                }
                System.out.println(Thread.currentThread().getName()+" Received:"+message);
                System.out.println(Thread.currentThread().getName()+" finished");
            //}
        });
        Thread thread2 = new Thread(()->{
            //synchronized (mylock){
                try {
                    Thread.sleep(n+1000);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                message = "Unipi!";
                //mylock.notify();
            //}
        });
        thread1.start();
        thread2.start();
    }
}
