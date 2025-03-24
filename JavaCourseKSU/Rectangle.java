// Job King 12/20/2018
// Program that prompts for length and width of a rectangle and calculates perimeter and area
// then displays perimeter and area
package edu.kennesaw.IT5413;
import java.util.Scanner;

public class Rectangle {

	public static void main(String[] args) {
	  Scanner scan = new Scanner(System.in); //declare input scanner
	  float len, win, perim, area; //declare floating point variables
	  
	  len=win=perim=area=0.00f; // initialize integer variables
	  
	  System.out.print("Enter rectangle length: "); //prompt for first floating point number rectangle length
	  len = scan.nextFloat();                     // read first floating point number from input rectangle length
	  
	  System.out.print("Enter rectangle width: "); // prompt for second floating point number rectangle width
	  win = scan.nextFloat();                    // read floating point number from input rectangle width
       
	  perim = 2*len + 2*win; // calculate rectangle perimeter
      area = len * win;      // calculate rectangle area
	  
        /** Print output */
	  
		System.out.printf("Perimeter is: %.3f\n", perim);  // print perimeter
        System.out.printf("Area is: %.3f\n", area);  // print area
		
		
		scan.close();

	}

}