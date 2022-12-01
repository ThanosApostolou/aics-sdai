package com.unipi.talepis.protal20221;

import java.util.function.Function;
import java.util.function.Predicate;

public class Demo2 {
    public static void main(String[] args) {
        Function<Integer, Integer> doubleInt = v -> 2*v;
        System.out.println("Result :"+doubleInt.apply(5));
        Predicate p = v->v!=null;
        Secretary s = null;
        System.out.println(p.test(new Student()));
        System.out.println(p.test(s));
    }
}
