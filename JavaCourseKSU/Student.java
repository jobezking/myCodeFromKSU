
public class Student {
	//Instance variables
	private String First_Name;
	private String Last_Name;
	private String Email;
   char Internship_compensation;
	char Internship_type;
	double GPA;
	int Student_year;
	
	//default constructor
	
	public Student()
	{
		//fill in the code to set default values to the instance variables
		First_Name = "";
		Last_Name = "";
		Email = "";
		Internship_compensation ='\u0000';
      Internship_type = '\u0000';
		GPA = 0.00;
		Student_year = 0;
	}
	
	// true Constructor
	
	public Student(String first, String last, String email, char e, char type, double gpa, int year)
	{
		//fill in the code to set default values to the instance variables
		First_Name = first;
		Last_Name = last;
		Email = email;
		Internship_compensation = e;
      Internship_type = type; 
		GPA = gpa;
		Student_year = year;
	}
	
	//Set Methods
	
	public void setFirst_Name(String x)	{
		First_Name = x;  
	}
	
	public void setLast_Name(String x)	{
		Last_Name = x;  
	}
	
	public void setEmail(String x)	{
		Email = x;  
	}
   
   public void setInternship_compensation (char x) {
      Internship_compensation = x;
   }
   
   public void setInternship_type(char q)	{
		Internship_type = q;  
	}
	public void setGPA(double z)	{
		GPA = z;  
	}
	
	public void setStudent_year(int y)	{
		Student_year= y;  
	}
	
	//Get Methods
	
	public String getFirst_Name()	{
		return First_Name;  
	}
	
	public String getLast_Name()	{
		return Last_Name;  
	}
	
	public String getEmail()	{
		return Email;  
	}
	public char getInternship_type()	{
		return Internship_type;  
	}
	
   public char getInternship_compensation() {
      return Internship_compensation;
    }//
   
   public double getGPA()	{
		return GPA;  
	}
	
	public int getStudent_year()	{
		return Student_year;  
	}
	
	//Other Methods i.e. toString
	
	// return String representation of Student
	public String toString()
	{
	  return String.format( "%s \t%s \t%s \t%c \t%s \t%.2f \t%d\n", 
			  First_Name, 
			  Last_Name, 
			  Email, 
			  Internship_type,
           Internship_compensation, //test for \t%s
			  GPA, 
			  Student_year);
	} //
	

}
