package com.unipi.talepis.protal20222.immutable;

import java.util.ArrayList;
import java.util.List;

public class DemoImmutable {
    public static void main(String[] args) {
        PostgraduateStudent Alexandra = new PostgraduateStudent("Alexandra",new Address("Greece","Piraeus"),
                null);
        /*Student s1 = (Student) Alexandra;
        System.out.println(s1.getName());
        Alexandra.doTheTrick("Iason");
        System.out.println(s1.getName());*/
        Address address1 = new Address("Greece","Piraeus");
        List<String> someCourses = new ArrayList<>();
        someCourses.add("Java");
        someCourses.add("Python");
        Student s2 = new Student("Evangelos",address1,someCourses);
        System.out.println(s2);
        address1.setCity("Athens");
        address1.setCountry("Italy");
        someCourses.remove("Java");
        someCourses.add("Android");
        System.out.println(s2);
        Address temp = s2.getAddress();
        temp.setCountry("USA");
        s2.getCourses().remove("Python");
        System.out.println(s2);
    }
}
