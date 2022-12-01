package com.unipi.talepis.protal20222.immutable;

import java.util.List;

public class PostgraduateStudent /*extends Student*/{
    private String hackStudentName;
    public PostgraduateStudent(String name, Address address, List<String> courses) {
        //super(name, address, courses);
        hackStudentName = name;
    }

    public void doTheTrick(String newName){
        hackStudentName = newName;
    }

    //@Override
    public String getName() {
        return hackStudentName;
    }
}
