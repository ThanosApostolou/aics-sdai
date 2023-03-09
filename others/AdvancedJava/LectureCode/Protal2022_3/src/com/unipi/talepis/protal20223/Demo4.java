package com.unipi.talepis.protal20223;

public class Demo4 {
    public static void main(String[] args) {
        Thread.startVirtualThread(new Runnable() {
            @Override
            public void run() {
                System.out.println("Hello "+Thread.currentThread().getName());
            }
        });
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
