// TestPrimeCounter.java
// Test excution time in ms for PrimeConter.java
// > java TestPrimeCounter 1000000
import java.util.Date;

public class TestPrimeCounter {
   public static void main (String[] args) { 
      Date start = new Date();   // JDK 1.0+

      // Do something by tested task
      PrimeCounter.main( args ); // pass args to the main method

      Date end = new Date();
      long timeInMS = end.getTime() - start.getTime(); 

      System.out.printf( "Run PrimeConter with %s elapsed %d ms.\n",
         args[0], timeInMS
      );
   }   
}
