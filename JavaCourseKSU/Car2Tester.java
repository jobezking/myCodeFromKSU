public class Car2Tester
{
   public static void main (String [] args)
   {
      //create an object of the car2 class
      Car2 myCar= new Car2();
      
      //parameterized constructor method called
      Car2 kCar = new Car2("Toyota", "Camry", 1234);
      Car2 sCar = new Car2("Honda", "Civic", 2345);
      myCar.setMake("Ford");
      
      
      System.out.println("make:  " + myCar.getMake());
      System.out.println("make:  " + kCar.getMake());
      System.out.println("make:  " + sCar.getMake());
      
      
      //display 
      System.out.println(myCar.display());
      System.out.println(kCar.display());
    }
 }