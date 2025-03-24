package edu.kennesaw.IT5413;
import java.util.Scanner;

public class Calculate {

	public static void main(String[] args) {
	  Scanner scan = new Scanner(System.in); //declare input scanner
	  int num1, num2, sum, prod, diff, quot; //declare integer variables
	  
	  sum=quot=prod=diff=num1=num2=0; // initialize integer variables
	  
	  System.out.print("Enter first integer: "); //prompt for first integer
	  num1 = scan.nextInt();                     // read first integer from input
	  
	  System.out.print("Enter next integer: "); // prompt for next integer
	  num2 = scan.nextInt();                    // read 2nd integer from input
	  
	  sum = num1 + num2;       // sum integers
	  prod = num1 * num2;      // multiply integers
	  diff = num1 - num2;      // subtract integers
      quot = num1 / num2;      // divide integers
        
        /** Print output */
	  
		System.out.printf("The sum is: %d\n", sum);
		System.out.printf("The product is: %d\n", prod);
        System.out.printf("The difference is: %d\n", diff);
        System.out.printf("The quotient is: %d\n", quot);
		
	  scan.close();

	}

}