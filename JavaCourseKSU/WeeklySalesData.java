/* Name: Job King
*  Date: April 7, 2018
*  Assignment 7-1 
* Software Design & Development Section W01 Spring Semester 2018 CO
* Write a program using an array that will process sales data for a week. 
* Create an array that will store input data for the daily sales for a week (7 days). 
* Using the array data determine the total sales for the week, 
* the day of the week with the least number of sales 
* and the day of the week with the highest sales.
* 
*/

import java.util.Scanner; // scanner class to parse input


public class WeeklySalesData {  // class declaration

	public static void main(String[] args) { // main class
		
		Scanner scan = new Scanner(System.in); //declare input scanner
		int loopctrl;                          // loop control variable
		float[] weeklysales = new float[7];		// define floating point array of 7 members
		float total=0;							// for total array value						
		float highest=0;                        // for highest in array
		float lowest=0;							// for lowest in array
		
		for (loopctrl = 0; loopctrl < weeklysales.length; loopctrl++)	//to initialize array with values 1-7
		{
			weeklysales[loopctrl] = (float)loopctrl+1;  // not necessary in Java according to text fig 7.2 ppg 244-5
		}
		
		for (loopctrl = 0; loopctrl < weeklysales.length; loopctrl++) // prompts user to fill each array element
		{// with floating point values for each sales day
			System.out.printf("Enter sales data for day %d : \n", loopctrl); // prompt message
			weeklysales[loopctrl] = scan.nextFloat(); // read sales day value from input
		}
		
		lowest = weeklysales[0]; // the highest value in array so far is the first value
		highest = lowest; // the lowest value in the array so far is the same as the highest
		
		for (loopctrl = 0; loopctrl < weeklysales.length; loopctrl++)
		{
			total = total + weeklysales[loopctrl]; // accumulate total
			
			if (weeklysales[loopctrl] > highest) // if current array value is higher than previous
				{
					highest = weeklysales[loopctrl]; // it is the new highest value for now
				}
			
			if (weeklysales[loopctrl] < lowest) // if current array value is lower than previous
				{
					lowest = weeklysales[loopctrl]; // it is the new lowest value for now
				}
		}		
		
		System.out.printf("Total - %.2f Highest - %.2f  Lowest - %.2f", total, highest, lowest); // print total, highest, lowest to screen
		
		scan.close(); // close input scanner object to prevent memory leak
	}

}
