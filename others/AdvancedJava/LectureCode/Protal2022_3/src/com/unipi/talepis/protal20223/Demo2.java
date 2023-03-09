package com.unipi.talepis.protal20223;

public class Demo2 {
    static Object o = new Object();
    public static void main(String[] args) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    System.out.println(Thread.currentThread().getName()+" is paused");
                    synchronized (o){
                        o.wait();
                    }
                    System.out.println("I am notified");
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }).start();
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    System.out.println(Thread.currentThread().getName()+" is paused");
                    synchronized (o){
                        o.wait();
                    }
                    System.out.println("I am notified");
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }).start();
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        synchronized (o){
            o.notify();
            o.notify();
        }
    }
}
