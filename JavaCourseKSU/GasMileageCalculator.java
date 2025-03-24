// Job King	2/17/2018
// Drivers are concerned with the gas mileage that their automobiles get. One driver has kept
// track of several trips by recording the miles driven and gallons used for each tankful.
// Develop a Java application that will input the miles driven and gallons used (both as integers)
// for each trip. The program should calculate and display the miles per gallon obtained for 
// each trip and print the combined miles per gallon obtained for all trips up to this point.
// All averaging calculations should produce floating point results. Use class Scanner and 
// sentinel-controlled repetition to obtain the data from the user

/* Complete exercise 4.17 on page 146 to calculate gas mileage. 
Run your program twice using different input. Copy the output and place it at the bottom of 
the program class as comments OR copy the output to a Word document and submit with the 
.java file and .class file.

This program can be written without writing a user-defined class.  
The problem can be solved with a class and main method. */

import java.util.Scanner;

public class GasMileageCalculator {

	public static void main(String[] args) {
	  Scanner scan = new Scanner(System.in); //declare input scanner
	  int trip_miles, trip_gallons, trips;
	  float trip_mpg, mpg_accum, mpg_avg;
	  trip_miles=trip_gallons=trips=0;
	  trip_mpg=mpg_accum=mpg_avg=0;
	  
	  System.out.print("\nEnter Trip Miles Driven or -1 to quit: ");
	  trip_miles = scan.nextInt();
	  
	  while (trip_miles != -1)
	  {
		  trips++;
		  System.out.print("Enter Trip Gallons: ");
		  trip_gallons = scan.nextInt();
		  trip_mpg = trip_miles / trip_gallons;
		  mpg_accum = mpg_accum + trip_mpg;
		  mpg_avg = mpg_accum / trips;
		  
		  System.out.printf("\nMiles per gallon for trip %d is %.2f\n", trips, trip_mpg);
		  System.out.printf("\nTotal Miles per gallon %.2f\n", mpg_accum);
		  System.out.printf("\nAverage Miles per gallon %.2f\n", mpg_avg);
		  
		  trip_miles=trip_gallons=0;
		  System.out.print("\nEnter Trip Miles Driven or -1 to quit: ");
		  trip_miles = scan.nextInt();
	  }
	  
		scan.close();

	}//public static void main(String[] args)

}//GasMileageCalculator

/* 
 /* Output 1 Enter Trip Miles Driven or -1 to quit: 10
Enter Trip Gallons: 2

Miles per gallon for trip 1 is 5.00

Total Miles per gallon 5.00

Average Miles per gallon 5.00
Enter Trip Miles Driven or -1 to quit: 100
Enter Trip Gallons: 10

Miles per gallon for trip 2 is 10.00

Total Miles per gallon 15.00

Average Miles per gallon 7.50
Enter Trip Miles Driven or -1 to quit: 8
Enter Trip Gallons: 1

Miles per gallon for trip 3 is 8.00

Total Miles per gallon 23.00

Average Miles per gallon 7.67
Enter Trip Miles Driven or -1 to quit: 300
Enter Trip Gallons: 8

Miles per gallon for trip 4 is 37.00

Total Miles per gallon 60.00

Average Miles per gallon 15.00
Enter Trip Miles Driven or -1 to quit: 1
Enter Trip Gallons: 15

Miles per gallon for trip 5 is 0.00

Total Miles per gallon 60.00

Average Miles per gallon 12.00
Enter Trip Miles Driven or -1 to quit: 66
Enter Trip Gallons: 5

Miles per gallon for trip 6 is 13.00

Total Miles per gallon 73.00

Average Miles per gallon 12.17
Enter Trip Miles Driven or -1 to quit: -1*/

/* Output 2 
 Enter Trip Miles Driven or -1 to quit: 33
Enter Trip Gallons: 2

Miles per gallon for trip 1 is 16.00

Total Miles per gallon 16.00

Average Miles per gallon 16.00

Enter Trip Miles Driven or -1 to quit: 500
Enter Trip Gallons: 15

Miles per gallon for trip 2 is 33.00

Total Miles per gallon 49.00

Average Miles per gallon 24.50

Enter Trip Miles Driven or -1 to quit: 1100
Enter Trip Gallons: 45

Miles per gallon for trip 3 is 24.00

Total Miles per gallon 73.00

Average Miles per gallon 24.33

Enter Trip Miles Driven or -1 to quit: 22
Enter Trip Gallons: 3

Miles per gallon for trip 4 is 7.00

Total Miles per gallon 80.00

Average Miles per gallon 20.00

Enter Trip Miles Driven or -1 to quit: 2
Enter Trip Gallons: 8

Miles per gallon for trip 5 is 0.00

Total Miles per gallon 80.00

Average Miles per gallon 16.00

Enter Trip Miles Driven or -1 to quit: 1
Enter Trip Gallons: 1000

Miles per gallon for trip 6 is 0.00

Total Miles per gallon 80.00

Average Miles per gallon 13.33

Enter Trip Miles Driven or -1 to quit: 1000
Enter Trip Gallons: 1

Miles per gallon for trip 7 is 1000.00

Total Miles per gallon 1080.00

Average Miles per gallon 154.29

Enter Trip Miles Driven or -1 to quit: 16
Enter Trip Gallons: 15

Miles per gallon for trip 8 is 1.00

Total Miles per gallon 1081.00

Average Miles per gallon 135.13

Enter Trip Miles Driven or -1 to quit: 15
Enter Trip Gallons: 16

Miles per gallon for trip 9 is 0.00

Total Miles per gallon 1081.00

Average Miles per gallon 120.11

Enter Trip Miles Driven or -1 to quit: -1
*/