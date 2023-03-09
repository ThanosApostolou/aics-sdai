package com.unipi.talepis.protal20223;

public class Demo1 {
    static String message;
    public static void main(String[] args) {
        Thread thread1 = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException e) {
                    System.out.println("I was interrupted...");
                    System.out.println("Do something with "+message);
                    message = "processed";
                }
            }
        });
        //thread1.setDaemon(true);
        thread1.start();
        //System.out.println("Finished!...");
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        message = "Hello unipi from "+Thread.currentThread().getName();
        System.out.println("Trying to interrupt thread1");
        thread1.interrupt();
        System.out.println("Main finished");
    }
}
