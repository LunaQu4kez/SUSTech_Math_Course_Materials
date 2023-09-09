// TestAllInOne.java
// Test excution time in ms for CounterPrimes.java & PrimeConter.java
// through TestCountPrimes.java & TestPrimeCounter
// by a series of parameter given by testCases. 
// > java TestAllInOne
// > java TestAllInOne > TestAllInOne-result.txt

import java.util.Date;

public class TestAllInOne {
   public static void main (String[] args) { 
      String[][] testCases = {
         { "1000" }, { "10000" }, { "100000" }, { "1000000" }, { "10000000" }
      };

      for (int i = 0; i < testCases.length; ++i) {
         TestCountPrimes.main( testCases[i] );
         TestPrimeCounter.main( testCases[i] );
      }
   }
}
