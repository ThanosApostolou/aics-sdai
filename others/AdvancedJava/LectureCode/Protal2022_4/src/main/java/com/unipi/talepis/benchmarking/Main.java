package com.unipi.talepis.benchmarking;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class Main {
    public static void main(String[] args) throws IOException, RunnerException {
        /*org.openjdk.jmh.Main.main(args);*/
        Options options = new OptionsBuilder()
                .include(com.unipi.talepis.Main.class.getSimpleName())
                .forks(1)
                .threads(3)
                .build();
        new Runner(options).run();
    }

    @Benchmark
    public void hi(){
        //do something!
    }
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void hello() throws InterruptedException {
        TimeUnit.MILLISECONDS.sleep(10);
    }
}
