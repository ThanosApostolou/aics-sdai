package com.unipi.talepis.protal20223;

public class Main {
    public static void main(String[] args) {
        Thread thread2 = new Thread(){
            @Override
            public void run() {
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        };
        thread2.start();
        try {
            thread2.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        /*try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }*/
        /*Thread thread1 = new Thread(new Runnable() {
            @Override
            public void run() {
                //System.out.println("Hi from "+Thread.currentThread().getName());
            }
        });
        thread1.start();*/
        //System.out.println("Hi from "+Thread.currentThread().getName());
        new Thread(()->printChar('A')).start();
        new Thread(()->printChar('B')).start();
        new Thread(()->printChar('C')).start();
    }
    static void printChar(char c){
        for (int i=0;i<100;i++){
            System.out.print(c);
        }
    }
}