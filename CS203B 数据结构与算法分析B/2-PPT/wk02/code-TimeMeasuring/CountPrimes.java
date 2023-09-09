public class CountPrimes { 
   public static void main (String[] args) {
      int N = Integer.parseInt( args[0] );
      int count = numberOfPrimes( N );
      System.out.printf( "%d Primes <= %d.\n", count, N );
   }
   public static int numberOfPrimes (int n) {
      boolean[] isPrime = seive( n+1 );
      int count = 0;
      for (int i = 2; i <= n; i++)
         if (isPrime[i]) count++;
      return count;
   }
   public static boolean[] seive (int n) {
      boolean[] b = new boolean[n];  
      for (int i = 2; i < n; i++)
         b[i] = true;
         
      int maxFactor = (int)(Math.sqrt(n) + 0.1);
      for (int i = 2; i <= maxFactor; i++) 
         if (b[i]) 
            for (int j = i*i; j < n; j += i)
               b[j] = false;
      
      return b;
   }
}
