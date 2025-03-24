/*Create an application that will accept sales information and calculate the total commission 
 * for a sales associate.  Each sale can is entered until the sentinel value is entered indicating 
 * the last sale.

There are 5 products which can be sold:

Product 1 with price of $6.98

Product 2 with price of $7.50

Product 3 with price of $3.75

Product 4 with price of $2.59

Product 5 with price of $23.79

The application should use a sentinel controlled loop to prompt the user for each product and quantity 
for a sales rep
Use a switch statement to identify the correct item price
Calculate the total sales  for all items for that sales rep and the commission of 5%.
Submit your .java file, your .class file and your screenshot testing your program.*/

import java.util.Scanner;

public class CalculateCommission {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in); //declare input scanner
		int product, poll;
		product=poll=0;
		double price,price_total, commission;
		price=price_total=commission=0;
		
		  System.out.print("\nEnter Product or -1 to quit: ");
		  product = scan.nextInt();
		  
		  while (product != -1)
		  {
			    poll++;
			    
			     switch(product)
			     {
			        case 1:
			        	price=6.98;
			        	break;
			        case 2:
			        	price=7.50;
			        	break;
			        case 3:
			        	price=3.75;
			        	break;
			        case 4:
			        	price=2.59;
			        	break;
			        default:
			        	price=23.79;
			      }
			     
			     price_total=price_total+price;
			  
			  System.out.printf("\nItem %d is Product %d Whose Price is %.2f\n", poll, product, price);
			  
			  product=0;
			  price=0;
			  System.out.print("\nEnter Product or -1 to quit: ");
			  product = scan.nextInt();
		  }
		  
	    commission = price_total * 0.05f;
		System.out.printf("\nTotal sales are %.2f and commission is %.2f\n", price_total, commission);
		scan.close();

	}
	
  
}
