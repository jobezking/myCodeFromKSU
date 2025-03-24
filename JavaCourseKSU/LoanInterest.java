// Job King	2/17/2018
/*Assignment 4-1 - DueMar 2, 2018 11:30 PM
Software Design & Development Section W01 Spring Semester 2018 CO
Determine Loan Interest Rate based on credit scores. Use if..else statements to find the interest
 rate.
Write a program that will give the interest rate for a new car loan based on a credit score. 

Credit score Interest Rate
850-720 5.56%
719-690 6.38%
660 689 7.12%
625-659 9.34%
590-624 12.45%
Below 590 no credit issued.

Prompt the user for the credit score.  Determine and display the user input credit score and the interest rate associated with the credit score.

Submit your .java, .class and screenshots as a zip file in this D2L link. */

import java.util.Scanner;
public class LoanInterest {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in); //declare input scanner
		int credit_score = 0;
		double interest_rate=0;
		
		System.out.print("\nEnter Credit Score: ");
		credit_score = scan.nextInt();
		
		if (credit_score >= 720)
		{
			interest_rate = 5.56;
		}
		else if (credit_score >= 690)
		{
			interest_rate = 6.38;
		}
		else if (credit_score >= 660)
		{
			interest_rate = 7.12;
		}
		else if (credit_score >= 625)
		{
			interest_rate = 9.34;
		}
		else if (credit_score >= 590)
		{
			interest_rate = 12.45;
		}
		
		System.out.printf("\nInterest rate for credit score %d is %.2f\n", credit_score, interest_rate);
		
		scan.close();

	}

}
