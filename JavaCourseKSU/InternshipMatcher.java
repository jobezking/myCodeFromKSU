 import java.util.Scanner;
 import java.util.ArrayList;

 
public class InternshipMatcher {

	public static void main(String[] args) {
        Scanner internshipEntry = new Scanner (System.in);
        int menu_option = 0;
        
        ArrayList <Student> studentList = new ArrayList <Student>(); 
        Student a = new Student ("John", "Brown", "johnbrown@yahoo.com", 'F','F', 3.5, 3); 
        Student b = new Student ("Erin", "Hayes", "hayes_erin@hotmail.com",'F', 'P', 2.3, 1); 
        Student c = new Student ("Aaron", "Cho", "chorun@atlanta.k12.ga.us", 'F', 'F', 2.8, 4); 
        Student d = new Student ("Harold", "Moore", "hmoore12345@gmail.com", 'F','P', 2.8, 2); 
        Student e = new Student ("Felicia", "Tyre", "ftyre@comcast.net", 'F','F', 4.0, 5); 
        studentList.add (a);     
        studentList.add (b); 
        studentList.add (c); 
        studentList.add (d); 
        studentList.add (e); 
        
        ArrayList <Company> companyList = new ArrayList <Company> (); 
        Company aa = new Company("Merck", "Jane Austen", "jane.austen@authors.net", "analyst", 'F', 
        		'P', 3, 3.5);
        Company bb = new Company("Comcast", "Walter Hayes", "walter.hayes@comcast.com", "developer", 'F', 
        		'P', 3, 2.5);
        Company cc = new Company("Google", "Mary Shelley", "frankenstein@prometheus.org", "editor", 'F', 
        		'P', 3, 3.0);
        Company dd = new Company("Amazon", "Janet Van Dyne", "thewasp@antman.com", "developer", 'F', 
        		'P', 3, 3.0);
        Company ee = new Company("Walgreens", "Hank Pym", "secret.identity@walgreens.com", "tech", 'F', 
        		'P', 3, 3.0);
        companyList.add(aa);
        companyList.add(bb);
        companyList.add(cc);
        companyList.add(dd);
        companyList.add(ee);
        
    	System.out.print( "\nSELECT MENU OPTION 1-9"); 
    	System.out.print( "\nMENU OPTION 1: PRINT ALL STUDENTS"); 
    	System.out.print( "\nMENU OPTION 2: PRINT ALL COMPANIES"); 
    	System.out.print( "\nMENU OPTION 3: MATCH STUDENTS TO INTERNSHIPS"); 
    	System.out.print( "\nMENU OPTION 4: ADD STUDENT"); 
    	System.out.print( "\nMENU OPTION 5: ADD COMPANY"); 
    	System.out.print( "\nMENU OPTION 6: DELETE STUDENT"); 
    	System.out.print( "\nMENU OPTION 7: DELETE COMPANY"); 
    	System.out.print( "\nMENU OPTION 8: UPDATE STUDENT"); 
    	System.out.print( "\nMENU OPTION 9: UPDATE COMPANY"); 
      System.out.println( "\nEnter your Selection:");
        
        menu_option = internshipEntry.nextInt();
        
        switch ( menu_option )
        {
        case 1: //  
        	PrintStudents(studentList);
        	break;
        case 2: // 
        	PrintCompanies(companyList);
        	break; //  
        case 3: //  
        	CompanyMatchStudentsToInternships(studentList, companyList);
        	break; 
        case 4: //  
        	addStudent(studentList);
        	break; //  
        case 5: //  
        	addCompany(companyList);
        	break; //  
        case 6: //  
        	deleteStudent(studentList);
        	break; //  
        case 7: //  
        	deleteCompany(companyList);
        	break; //  
        case 8: //  
        	updateStudent(studentList);
        	break; //  
        case 9: //  
        	updateCompany(companyList);
        	break; //  
        default: //  
        	System.out.print( "Invalid Selection"); 
        	break; // optional; will exit switch anyway
        } // end switch
        
           
        
        internshipEntry.close();

	}//end main
	
static void addStudent(ArrayList <Student> students)
   {
		
		Scanner newstudent = new Scanner (System.in);
		Student e = new Student();
		
		System.out.print( "ENTER STUDENT FIRST NAME: ");
		String fname = newstudent.next();
		e.setFirst_Name(fname);
		
		System.out.print( "ENTER STUDENT LAST NAME: ");
		String lname = newstudent.next();
		e.setLast_Name(lname);
		
		System.out.print( "ENTER STUDENT EMAIL ADDRESS: ");
		String mail = newstudent.next();
		e.setEmail(mail);
		
		System.out.print( "ENTER STUDENT INTERNSHIP TYPE: ");
		char interntype = newstudent.next().charAt(0);
		e.setInternship_type(interntype);
		
		System.out.print( "ENTER STUDENT GPA: ");
		double gp = newstudent.nextDouble();
		e.setGPA(gp);
		
		System.out.print( "ENTER STUDENT YEAR: ");
		int studentyear = newstudent.nextInt();
		e.setStudent_year(studentyear);
		
		students.add(e);
		
		newstudent.close();
	}

static void addCompany(ArrayList <Company> companies)
	{
	
		Scanner newcompany = new Scanner (System.in);
		Company e = new Company();
	
		System.out.print( "ENTER COMPANY NAME: ");
		String comp_name = newcompany.next();
		e.setCompany_name(comp_name);
	
		System.out.print( "ENTER CONTACT NAME: ");
		String contname = newcompany.next();
		e.setContact_name(contname);
	
		System.out.print( "ENTER CONTACT EMAIL ADDRESS: ");
		String commail = newcompany.next();
		e.setContact_email(commail);
	
		System.out.print( "ENTER INTERNSHIP TITLE: ");
		String interntitle = newcompany.next();
		e.setInternship_title(interntitle);
	
		System.out.print( "ENTER INTERNSHIP COMPENSATION: ");
		char compensation = newcompany.next().charAt(0);
		e.setInternship_compensation(compensation);
	
		System.out.print( "ENTER INTERNSHIP TYPE: ");
		char interntype = newcompany.next().charAt(0);
		e.setInternship_type(interntype);
	
		System.out.print( "ENTER REQUIRED STUDENT YEAR: ");
		int studentyear = newcompany.nextInt();
		e.setrequired_student_year(studentyear);
	
		System.out.print( "ENTER REQUIRED STUDENT GPA: ");
		double gp = newcompany.nextDouble();
		e.setrequired_student_GPA(gp);
	
		companies.add(e);
	
		newcompany.close();	
	
	}

static void deleteStudent(ArrayList <Student> students)
	{
		Scanner deleteStudent = new Scanner (System.in);
		System.out.print( "EMAIL ADDRESS OF STUDENT TO DELETE: ");
		String delete_mail = deleteStudent.next();
		//find student in array using email address
		for ( int i = 0; i < students.size(); i++ )
			{
			 if (students.get(i).getEmail().equalsIgnoreCase(delete_mail)) 
			   {
				 System.out.printf( "Student Deleted: %s", students.get( i ).toString() );
				 students.remove(i);
			   } 
			}
		
		deleteStudent.close();
	}

static void deleteCompany(ArrayList <Company> companies)
	{
		Scanner deleteCompany = new Scanner (System.in);
		System.out.print( "EMAIL ADDRESS OF CONTACT OF COMPANY TO DELETE: ");
		String delete_mail = deleteCompany.next();
		//find company in array using email address
		for ( int i = 0; i < companies.size(); i++ )
		{
		 if (companies.get(i).getContact_email().equalsIgnoreCase(delete_mail)) 
		   {
			 System.out.printf( "Company Deleted: %s", companies.get( i ).toString() );
			 companies.remove(i);
		   } 
		}
			
		deleteCompany.close();
	}

static void updateStudent(ArrayList <Student> students)
   {
	Scanner updateStudent = new Scanner (System.in);
   
	System.out.print( "EMAIL ADDRESS OF STUDENT TO UPDATE: ");
	String update_mail = updateStudent.next();
	for ( int i = 0; i < students.size(); i++ )
	{
	 if (students.get(i).getEmail().equalsIgnoreCase(update_mail)) 
	   {
		 System.out.printf( "Student To Update: %s", students.get( i ).toString() );
		 
			System.out.print( "ENTER STUDENT FIRST NAME: ");
			String fname = updateStudent.next();
			students.get(i).setFirst_Name(fname);
			
			System.out.print( "ENTER STUDENT LAST NAME: ");
			String lname = updateStudent.next();
			students.get(i).setLast_Name(lname);
			
			students.get(i).setEmail(update_mail);
			
			System.out.print( "ENTER STUDENT INTERNSHIP TYPE: ");
			char interntype = updateStudent.next().charAt(0);
			students.get(i).setInternship_type(interntype);
			
			System.out.print( "ENTER STUDENT GPA: ");
			double gp = updateStudent.nextDouble();
			students.get(i).setGPA(gp);
			
			System.out.print( "ENTER STUDENT YEAR: ");
			int studentyear = updateStudent.nextInt();
			students.get(i).setStudent_year(studentyear);
			
			System.out.printf( "Student Updated: %s", students.get( i ).toString() );
	   } 
	}
	updateStudent.close();	
	}

static void updateCompany(ArrayList <Company> companies)
	{
	Scanner updateCompany = new Scanner (System.in);
	System.out.print( "EMAIL ADDRESS OF CONTACT OF COMPANY TO UPDATE: ");
	String update_email = updateCompany.next();
	for ( int i = 0; i < companies.size(); i++ )
	{
	 if (companies.get(i).getContact_email().equalsIgnoreCase(update_email)) 
	   {
		 System.out.printf( "Company To Update: %s", companies.get( i ).toString() );
			
			System.out.print( "ENTER COMPANY NAME: ");
			String comp_name = updateCompany.next();
			companies.get(i).setCompany_name(comp_name);
		
			System.out.print( "ENTER CONTACT NAME: ");
			String contname = updateCompany.next();
			companies.get(i).setContact_name(contname);
		
			companies.get(i).setContact_email(update_email);
		
			System.out.print( "ENTER INTERNSHIP TITLE: ");
			String interntitle = updateCompany.next();
			companies.get(i).setInternship_title(interntitle);
		
			System.out.print( "ENTER INTERNSHIP COMPENSATION: ");
			char compensation = updateCompany.next().charAt(0);
			companies.get(i).setInternship_compensation(compensation);
		
			System.out.print( "ENTER INTERNSHIP TYPE: ");
			char interntype = updateCompany.next().charAt(0);
			companies.get(i).setInternship_type(interntype);
		
			System.out.print( "ENTER REQUIRED STUDENT YEAR: ");
			int studentyear = updateCompany.nextInt();
			companies.get(i).setrequired_student_year(studentyear);
		
			System.out.print( "ENTER REQUIRED STUDENT GPA: ");
			double gp = updateCompany.nextDouble();
			companies.get(i).setrequired_student_GPA(gp);
			
			System.out.printf( "Company Updated: %s", companies.get( i ).toString() );
	   } 
	}
	updateCompany.close();	
	}

static void CompanyMatchStudentsToInternships(ArrayList <Student> students,  ArrayList <Company> companies)
   {
	for ( int i = 0; i < students.size(); i++ )
		{
		 System.out.printf( " %s", students.get( i ).toString() );
		}
	for ( int i = 0; i < companies.size(); i++ )
		{
		 System.out.printf( " %s", companies.get( i ).toString() );
		}
	}

static void PrintStudents(ArrayList <Student> students  ) 
	{
	for ( int i = 0; i < students.size(); i++ )
		{
		 System.out.printf( " %s", students.get( i ).toString() );
		}
	} 

static void PrintCompanies( ArrayList <Company> companies)  
	{
	for ( int i = 0; i < companies.size(); i++ )
		{
		 System.out.printf( " %s", companies.get( i ).toString() );
		}
	} //

} // end program