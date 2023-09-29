package aics;

import io.quarkus.logging.Log;
import io.quarkus.runtime.Quarkus;
import io.quarkus.runtime.annotations.QuarkusMain;

@QuarkusMain
public class Main {
    public static void main(String... args) {
        Log.trace("Start Main.main");
        Quarkus.run(TicketApp.class, args);
    }

}