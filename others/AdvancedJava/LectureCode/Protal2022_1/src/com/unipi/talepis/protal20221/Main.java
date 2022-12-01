package com.unipi.talepis.protal20221;

import java.util.ArrayList;
import java.util.List;

public class Main {
    static List<Student> students = new ArrayList<>();
    public static void main(String[] args) {
        Student s1 = new Student();
        s1.am = "mpsp22123";
        s1.name = "Alexandra";
        Student s2 = new Student();
        s2.am = "mpsp22007";
        s2.name = "Evangelos";
        students.add(s1);
        students.add(s2);
        Secretary katerina = new Secretary();
        katerina.printStudentList(students);
        System.out.println("---------------------------");
        katerina.printStudentList(students);
        System.out.println("##########################");
        System.out.println("Student name:"+s1.name);
        //Clear view of Reference types
        //Shallow copy Vs Deep Copy
        //Mutable objects Vs Immutable objects
    }
}