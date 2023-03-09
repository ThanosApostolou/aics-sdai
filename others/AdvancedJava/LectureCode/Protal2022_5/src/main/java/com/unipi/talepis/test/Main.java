package com.unipi.talepis.test;

public class Main {

        class Simple{

}
        interface My{
        }

        static class  Test4{
            public static void main(String[] args){
                try{
                    Class c = Class.forName("com.unipi.talepis.test.Student");
                    System.out.println(c.isInterface());
                    Class c2 = Class.forName("My");
                    System.out.println(c2.isInterface());
                    byte [] bytes = new byte [1024];
                    Class c3 = bytes.getClass();
                    System.out.println(c3.isArray());
                    Class c4 = boolean.class;
                    System.out.println(c4.isPrimitive());
                }catch(Exception e){
                    System.out.println("My name is Bled");

                }
            }
        }
    }
//}
class Student{

}


//public static void main(String[] args) {
       /* try {
            Class c = Class.forName("com.unipi.talepis.test.Student");
            System.out.println(c.getSimpleName());
            System.out.println(c.getName());

        } catch (Exception e) {

        }
        */