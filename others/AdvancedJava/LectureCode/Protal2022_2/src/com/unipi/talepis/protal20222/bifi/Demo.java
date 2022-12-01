package com.unipi.talepis.protal20222.bifi;

import java.util.function.*;

public class Demo {
    public static void main(String[] args) {
        Supplier<Integer> ageSupplier = ()->25;
        System.out.println(ageSupplier.get());
        Consumer<Integer> integerConsumer = integer -> System.out.println("Hello "+integer);
        integerConsumer.accept(ageSupplier.get());
        BiConsumer<String,Integer> biConsumer = ((s, integer) -> System.out.println(s+" "+integer));
        biConsumer.accept("Greetings",ageSupplier.get());
        Predicate<String> stringPredicate = s -> s.equals("Alepis");
        System.out.println(stringPredicate.test("Dimitris"));
        System.out.println(stringPredicate.test("Alepis"));
        Function<String,Boolean> myFunction = s -> s.startsWith("E");
        System.out.println(myFunction.apply("Evangelos"));
    }
}
