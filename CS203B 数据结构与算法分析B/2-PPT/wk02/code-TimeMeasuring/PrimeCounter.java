public class PrimeCounter {
   public static void main (String[] args) {
      int N = Integer.parseInt( args[0] );
      int count = numberOfPrimes( N );
      System.out.printf( "%d Primes in [1..%d]\n", count, N );
   }

   public static int numberOfPrimes (int n) {
      if (n <= 1) return 0;
      if (n == 2) return 1;
      
      int nPrimes = 1;
      for (int i = 3; i <= n; i = i+2) {
         boolean isPrime = true;
         int maxFactor = (int) (Math.sqrt( i ) + 0.1);
         for (int k = 3; k <= maxFactor; k = k+2) 
            if (i % k == 0) { isPrime = false; break; } 
         if (isPrime) nPrimes++;
      }
      
      return nPrimes;
   }
}
