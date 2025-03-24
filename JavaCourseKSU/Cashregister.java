package edu.kennesaw.IT5413;
import java.util.Scanner;

public class Cashregister {

	public static void main(String[] args) {
		  Scanner scan = new Scanner(System.in); //declare input scanner
		  float tax, subtotal, total, num1, num2, num3; //declare floating point variables
		  float tax_rate=0.06f;
          
		  tax=subtotal=total=num1=num2=num3=0.00f; // initialize integer variables
		  
		  System.out.print("Enter first value: "); //prompt for first floating point number
		  num1 = scan.nextFloat();                     // read first floating point number from input
		  
		  System.out.print("Enter second value: "); // prompt for second floating point number
		  num2 = scan.nextFloat();                    // read floating point number from input
	        
	      System.out.print("Enter third value: ");  // prompt for third floating point number
	      num3 = scan.nextFloat();                   // read third floating point number
	      
	      subtotal = num1 + num2 + num3;       // calculate subtotal
	      tax = subtotal * tax_rate;         // calculate total
	      total = subtotal + tax;       // calculate tax
	      
		System.out.printf("Item 1 is: %.2f\n",  num1);  // print first input
	    System.out.printf("Item 2 is: %.2f\n",  num2);  // print second input
	    System.out.printf("Item 3 is: %.2f\n", num3);  // print third input
	    System.out.printf("Subtotal is: %.2f\n", subtotal);  // print subtotal
	    System.out.printf("Tax is: %.2f\n",  tax);      // print tax
	    System.out.printf("Total is: %.2f\n", total);   // print total
		          
		  
		  scan.close();
	}

}