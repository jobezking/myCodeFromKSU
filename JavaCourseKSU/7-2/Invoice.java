/* Name: Job King
*  Date: February 8, 2018
* 
 \\ Create a class called Invoice that a hardware store might use to represent
 \\ an invoice for an item sold at the store. An Invoice should include four 
 \\ pieces of information as instance variables: a part number (type String),
 \\ a part description (type String), a quantity of item being purchased (type int)
 \\ and a price per item (double). Your class should have a constructor that
 \\ initializes the four instance variables. Provide a set and a get method for each
 \\ instance variable. In addition, provide a method named getInvoiceAmount that calculates
 \\ the invoice amount i.e. multiplies the quantity by the price per item then returns
 \\ the amount as a double value. If the quantity is not positive it should be set to 0.0.
 \\ Write a test application named InvoiceTest that demonstrates class Invoice's capabilities.
*/
public class Invoice
{
	//Instance variables
	private String Part_num;
	private String Part_desc;
	private int Num_purch;
	private double Item_price;
	
	//default constructor
	
	public Invoice()
	{
		//fill in the code to set default values to the instance variables
		Part_num ="";  // set first name to null
		Part_desc = "";  // set last name to null
		Num_purch = 0; // set salary to zero
		Item_price = 0;
	}
	
	//Constructor initializing instance variables with arguments
	public Invoice(String pnum, String pdesc, int numpurch, double itemprice)
	{
		Part_num = pnum;   
		Part_desc = pdesc;   
		Num_purch = numpurch; 
		validateItem_price(itemprice);			
	}
	
	private void validateItem_price(double x)
	{
		if (x > 0)  // check to see if Item_price is positive
		  {	
			Item_price=x;     // if so then set Item_price with argument
		  }  
		else
		    {
			Item_price=0;  // if not then Item_price should be zero
		    } 
		//complete	
	}
	
	
	public void setPart_num(String x)
	{
		Part_num = x;  // set part number
	}
	
	public void setPart_desc (String x)
	{
		Part_desc=x; //set part description
	}	
		
	
	public void setNum_purch (int x)
	{
		Num_purch = x; //set number of items s purchased
	}
	
	public void setItem_price (double x)  // make sure item price is positive
	{
		validateItem_price(x);
		
	}
	
	public String getPart_num()
	{
		return Part_num;  // return part number
	}
		
	//add the two additional methods getLast() and getSalary
	
	public String getPart_desc()
	{
		return Part_desc;   // return part description
	}
		
	
	public double getItem_price()
	{
		return Item_price;   // return item price
	}
	
	public int getNum_purch()
	{
		return Num_purch;   // return number of items purchased
	}
	
	
	public double getInvoiceAmount()
	{	
		return Num_purch * Item_price;  // calculate and return invoice amount
	}
}//end Class
	
	