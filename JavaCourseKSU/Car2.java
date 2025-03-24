public class Car2
{
   //declare instance variable for a car
   private String make;
   private String model;
   private int vin;
   
   //default constructor
   public Car2 (){
      make = "";
      model = "";
      vin = 0;
      }
   
   //parameterized constructor
   public Car2 (String m, String mo, int v)
   {
      make=m;
      model=mo;
      vin=v;
    }
    
    //set method - mutator methods
    public void setMake(String m){
      make=m;
    }
    //write other set methods for model and vin
    
    
    //get method - accessor methods
    public String getMake(){
      return make;
      }
   
   public String display(){
      String s;
      s = " " + make +"\t" + model + "\t" + vin;
      return s;
          
      }
   
   }