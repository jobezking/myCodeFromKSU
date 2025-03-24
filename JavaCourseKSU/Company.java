
public class Company {
	
	//Instance variables
	
	private String Company_name;
	private String Contact_name;
	private String Contact_email;
	private String Internship_title;
	char Internship_compensation;
	char Internship_type;
	int required_student_year;
	double required_student_GPA;
	
	//default constructor
	
	public Company()
	{
		//fill in the code to set default values to the instance variables
		Company_name = "";
		Contact_name = "";
		Contact_email = "";
		Internship_title = "";
		Internship_compensation = '\u0000';
		Internship_type = '\u0000';
		required_student_year = 0;
		required_student_GPA = 0.00f;
	}
	
	// true constructor
	
	public Company(String a, String b, String c, String d, char e, char f, int g, double h)
	{
		//fill in the code to set default values to the instance variables
		Company_name = a;
		Contact_name = b;
		Contact_email = c;
		Internship_title = d;
		Internship_compensation = e;
		Internship_type = f;
		required_student_year = g;
		required_student_GPA = h;
	}
	
	
	//Set Methods
	
	public void setCompany_name(String x)	{
		Company_name = x;  
	}
	
	public void setContact_name(String x)	{
		Contact_name = x;  
	}
	
	public void setContact_email(String x)	{
		Contact_email = x;  
	}
	public void setInternship_title(String x)	{
		Internship_title = x;  
	}
	
	public void setInternship_compensation(char x)	{
		Internship_compensation = x;  
	}
	
	public void setInternship_type(char w)	{
		Internship_type = w;  
	}
	
	public void setrequired_student_year(int y)	{
		required_student_year = y;  
	}
	
	public void setrequired_student_GPA(double z)	{
		required_student_GPA = z;  
	}
	
	// Get Methods
	
	public String getCompany_name()	{
		return Company_name;  
	}
	
	public String getContact_name()	{
		return Contact_name; 
	}
	
	public String getContact_email()	{
		return Contact_email;  
	}
	public String getInternship_title()	{
		return Internship_title;
	}
	
	public char getInternship_compensation()	{
		return Internship_compensation;  
	}
	
	public char getInternship_type()	{
		return Internship_type;  
	}
	
	public int getrequired_student_year()	{
		return required_student_year;  
	}
	
	public double getrequired_student_GPA()	{
		return required_student_GPA;
	}
	
	//Other Methods i.e. toString
	
	// return String representation of Company
	public String toString()
	{
		return String.format( "%s \t%s \t%s \t%s \t%s \t%c \t%.2f \t%d\n", 
				Company_name,
				Contact_name,
				Contact_email,
				Internship_title,
				Internship_compensation,
				Internship_type,
				required_student_GPA,
				required_student_year);
	} //

}
