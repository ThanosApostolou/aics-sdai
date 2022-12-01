package com.unipi.talepis.protal20221;

import java.util.ArrayList;
import java.util.List;

public class Secretary {
    void printStudentList(List<Student> sList){
        for(Student s : sList){
            System.out.println(s.name+","+s.am);
        }
        sList = new ArrayList<>();
        sList.add(new Student());
        //students.get(0).name = "Georgia";
    }
}
