package org.example;

import java.lang.annotation.*;
import java.lang.reflect.Method;

public class Main {

    public static void main(String[] args) {
        Class c = SomeClass.class;
        Annotation[] annotations = c.getAnnotations();
        for (Annotation annotation : annotations){
            if (annotation instanceof MyAnnotation){
                MyAnnotation myAnnotation = (MyAnnotation) annotation;
                System.out.println(myAnnotation.name());
                System.out.println(myAnnotation.description());
            }
        }
        try {
            Method method = c.getMethod("hello");
            Annotation[] methodAnnotations = method.getDeclaredAnnotations();
            for (Annotation annotation : methodAnnotations){
                if (annotation instanceof MyAnnotation){
                    MyAnnotation myAnnotation = (MyAnnotation) annotation;
                    System.out.println(myAnnotation.name());
                    System.out.println(myAnnotation.description());
                }
            }
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }
}

@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.CONSTRUCTOR,ElementType.METHOD,ElementType.TYPE})
@interface MyAnnotation{
    String name();
    String description();
}

@MyAnnotation(name = "class",description = "my first class")
class SomeClass{
    private String email;
    @MyAnnotation(name = "hello",description = "Very useful method")
    public void hello(){
        System.out.println("Hello Unipi!");
    }
}