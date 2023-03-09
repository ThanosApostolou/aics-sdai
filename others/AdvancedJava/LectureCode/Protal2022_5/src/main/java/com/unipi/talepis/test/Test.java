package com.unipi.talepis.test;

import java.util.ArrayList;
import java.util.List;

public class Test {
    public static void main(String[] args) {
        List<String> strings = new ArrayList<>();
        strings.add("Efthimios");
        System.out.println(strings);
        changeList(strings);
        System.out.println(strings);
        if (!strings.contains("Efthimios"))
            strings.add("Efthimios");
        System.out.println(strings);
    }
    static void changeList(List<String> strings){
        strings.add("Alepis");
    }
}
