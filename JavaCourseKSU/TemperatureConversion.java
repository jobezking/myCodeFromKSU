 // Fig. 6.3: TemperatureConversion.java
 // Programmer-declared method maximum with three double parameters.
 import java.util.Scanner;

public class TemperatureConversion
 {
 // obtain three floating-point values and locate the maximum value
 public static void main( String[] args )
 {
 // create Scanner for input from command window
 char celorfahr = 'Q';	 // determines temperature conversion
 int temp = 0; // temperature value
 Scanner scan = new Scanner( System.in );  // scan terminal for input object
 
 System.out.print("\nEnter 'F' for Celsius To Fahrenheit, 'C' for Fahrenheit To Celsius or 'Q' to quit: "); // user must decide
                  //Whether Celsuis, Fahrenheit or to Quit
 celorfahr = scan.next().charAt(0); // scans user option from input
 
 while (celorfahr != 'Q')  //continue until user is ready to stop by entering Q
 {
	     switch(celorfahr)
	     {
	        case 'C':  // if celsius
	       	 	System.out.print("\nEnter Integer Fahrenheit Temperature: "); // prompt user for Celsius
	       	 	temp = scan.nextInt();   // get Celsius value
	       	    System.out.printf("\nCelsius Temperature is: %d\n", ToCelsius(temp));
	        	break;
	        case 'F':
	       	 	System.out.print("\nEnter Integer Celsius Temperature: "); // prompt user for Fahrenheit
	       	 	temp = scan.nextInt();   
	       	 	System.out.printf("\nFahrenheit Temperature is: %d \n", ToFahrenheit(temp));
	        	break;
	        default:
	        	System.out.print("\nInput Error: Only 'F','C' or 'Q' are valid values."); //default error message
	      }
	     
	     System.out.print("\nEnter 'F' for Fahrenheit, 'C' for Celsius or 'Q' to quit: "); // user must decide
         //Whether Celsuis, Fahrenheit or to Quit
	     celorfahr = scan.next().charAt(0); // scans user option from input
 }

 // display maximum value
 System.out.println("Thank you for your participation in this exercise!" ); // ending progam message
 
 scan.close(); 
 } // end main
 
 static int ToCelsius( int fahrenheit )  // static int method to convert fahrenheit to celsius
 {
	 return (int)(5.0 / 9.0 * ( fahrenheit - 32 )); //perform temperature conversion and cast to int
 } // end method ToCelsius
 
 static int ToFahrenheit( int celsius )  // static int method to convert celsius to fahrenheit
 {
	 return (int)(9.0 / 5.0 * celsius + 32); //perform temperature conversion and cast to int
 } // end method ToFahrenheit
 
 } // end program