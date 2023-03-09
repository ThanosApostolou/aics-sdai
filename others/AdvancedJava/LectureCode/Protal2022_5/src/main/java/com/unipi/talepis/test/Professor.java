package com.unipi.talepis.test;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

class Human{
    int age;
    void walk(){
        System.out.println("Walking");
    }
}
interface ITeach{
    void teach();
    default void teachMore(){
        System.out.println("Teaching more!...");
    }
}
interface IEat{
    void eat();
}
public class Professor extends Human implements ITeach,IEat{
    private int officeNumber;
    protected static float secret;
    public Professor() {
    }

    public Professor(String name) {
        this.name = name;
    }

    public String name;

    private void evaluateTest(String test){
        System.out.println("Do work with "+test);
    }

    @Override
    public void teach() {
        System.out.println("Teaching Java!!");
    }

    @Override
    public void eat() {

    }
    public String test (String courseName){
        return "I am testing "+courseName;
    }
}

class DemoReflection{
    public static void main(String[] args) {
        Professor p1 = new Professor();
        Field[] fields = p1.getClass().getFields();
        List<String> fieldNames = Arrays.stream(fields)
                .map(field -> field.getName())
                .collect(Collectors.toList());
        System.out.println(fieldNames);
        try {
            Class professorClass = Class.forName("com.unipi.talepis.test.Professor");
            System.out.println(Modifier.toString(professorClass.getModifiers()));
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        Package p = Professor.class.getPackage();
        System.out.println(p.getName());
        Class<? super Professor> professorSuperClass = Professor.class.getSuperclass();
        System.out.println(professorSuperClass.getSimpleName());
        Class[] professorInterfaces = Professor.class.getInterfaces();
        for (Class c:professorInterfaces){
            System.out.println(c.getSimpleName());
        }
        Constructor<?>[] constructors = Professor.class.getConstructors();
        for (Constructor c: constructors){
            System.out.println(c.getName());
        }
        try {
            Method[] methods = Class.forName("com.unipi.talepis.test.Professor").getDeclaredMethods();
            for (Method m :
                    methods) {
                System.out.println(m.getName());
            }
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}


