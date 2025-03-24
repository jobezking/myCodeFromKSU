/* Name: Job King
*  Date: April 23, 2018
* Assignment 7-2: Arraylists
*  Create a tester class using your Invoice class from Chapter 3 (or see attached).  
*  Create an arraylist of 5 items.  Display part number, part description, 
*  price, quantity and total amount of each invoice.  
*  Also determine and display the average number of 
*  items ordered and the total of all invoices.
 \\ Write a test application named InvoiceTest 
 that demonstrates class Invoice's capabilities.
*/

import java.util.ArrayList;

public class InvoiceTester
{
	public static void main (String [] args)
	{
		// Create two instances of the Invoice class
		ArrayList <Invoice> Invoices = new ArrayList <Invoice>();
		Invoice _iv_1 = new Invoice();  // create first invoice
		double avg_ordered = 0;
		double invoice_total = 0;
		int i;
		
		// public Invoice(String pnum, String pdesc, int numpurch, double itemprice)
		// private String Part_num;
		// private String Part_desc;
		// private int Num_purch;
		// private double Item_price;
		
		
		
		Invoice _iv_2 = new Invoice("37", "hammer", 100, 1.0); // create second Invoice
		Invoice _iv_3 = new Invoice("2", "nail", 200, 2.0); // create third Invoice
		Invoice _iv_4 = new Invoice("3", "ladder", 300, 3.0); // fourth second Invoice
		Invoice _iv_5 = new Invoice("4", "bucket", 400, 4.0); // fifth second Invoice

		 
		_iv_1.setPart_num("7");  // set part number
		_iv_1.setPart_desc("saw");  // set part name
		_iv_1.setNum_purch(5);   // set number of parts bought
		_iv_1.setItem_price(5.0); // set part price
		//add the additional information
		 
		Invoices.add(_iv_1);  // add item to arraylist
		Invoices.add(_iv_2); // add item to arraylist
		Invoices.add(_iv_3); // add item to arraylist
		Invoices.add(_iv_4); // add item to arraylist
		Invoices.add(_iv_5);// add item to arraylist
		
		//display Invoice data
		
		for ( i = 0; i < Invoices.size(); i++ )
		{		
			System.out.printf("Invoice %d Part Number:  %s\n", i,Invoices.get(i).getPart_num());
			System.out.printf("Invoice %d Part Description:  %s\n", i,Invoices.get(i).getPart_desc());
			System.out.printf("Invoice %d Items Purchased:  %s\n", i,Invoices.get(i).getNum_purch());
			System.out.printf("Invoice %d Price of Item:  %s\n", i,Invoices.get(i).getItem_price());
			System.out.printf("Invoice %d Total Amount:  %s\n", i,Invoices.get(i).getInvoiceAmount());
			System.out.println(" ");
			
			avg_ordered = avg_ordered + Invoices.get(i).getNum_purch();  // compute cumulative order amount
			invoice_total = invoice_total + Invoices.get(i).getInvoiceAmount(); // cumulative Invoice amount
		}
		
		System.out.printf("Total Ordered:  %.2f	Total items:  %d \n", avg_ordered, i);
		avg_ordered = avg_ordered / i;
		System.out.printf("Average Ordered:  %.2f	Invoice Total:  %.2f ", avg_ordered, invoice_total);
		
		
	}//end method main
} //end class InvoiceTester

