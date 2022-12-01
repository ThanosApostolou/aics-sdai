package com.unipi.talepis.protal20222.immutable;

import java.util.ArrayList;
import java.util.List;


/*1. Remove Setters
        2. Add all args constructor to the class
3. Mark class as final to prevent inheritance
        4. Initialize all non-primitive mutable fields via constructor ONLY by performing deep copy(!!!)
        5. For all non-primitive mutable fields getter methods, perform cloning

        6. Mark all class attributes as final (optional, be careful)*/
public final class Student {
    private String name;
    private Address address;
    private List<String> courses;

    public Student(String name, Address address, List<String> courses) {
        this.name = name;
        this.address = new Address(address.getCountry(),address.getCity());
        this.courses = new ArrayList<>(courses);
    }

    public String getName() {
        return name;
    }

    public Address getAddress() {
        return new Address(address.getCountry(),address.getCity());
    }

    public List<String> getCourses() {
        return new ArrayList<>(courses);
    }

    @Override
    public String toString() {
        return "Student{" +
                "name='" + name + '\'' +
                ", address=" + address +
                ", courses=" + courses +
                '}';
    }
}
