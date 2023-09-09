// TestCountPrimes.java
// Test excution time in ms for CounPrimes.java
// > java TestCountPrimes 1000000
import java.time.Instant;
import java.time.Duration;

public class TestCountPrimes {
   public static void main (String[] args) { 
      Instant start = Instant.now(); // JDK 8.0+

      // Do something by tested task
      CountPrimes.main( args ); // pass args to the main method

      Instant end = Instant.now();
      long timeInMS = Duration.between(start, end).toMillis(); 

      System.out.printf( "Run CountPrimes with %s elapsed %d ms.\n",
         args[0], timeInMS
      );
   }   
}
