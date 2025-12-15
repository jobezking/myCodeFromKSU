class Calculator:
    def __init__(self, a, b, c):
        # store a, b, c as attributes of the object
        self.a = a
        self.b = b
        self.c = c
        self.value = 0   # optional: track the current result

    def add(self):
        self.value = self.a + self.b + self.c
        return self.value

    def subtract(self, x):
        self.value = self.value - x
        return self.value

    def square(self):
        self.value = self.value ** 2
        return self.value

    def output(self):
        print(self.value)


# Using the class
calc = Calculator(1, 2, 3)   # a=1, b=2, c=3
calc.add()                   # value = 6
calc.subtract(1)              # value = 5
calc.square()                 # value = 25
calc.output()                 # prints 25


# Using the class
calc = Calculator()
calc.add(1, 2, 3)     # value = 6
calc.subtract(1)      # value = 5
calc.square()         # value = 25
calc.output()         # prints 25

#Original
'''
def add(a, b, c): 
    return a+b+c 
    
def subtract(d,a): 
    return d-a 

def square(z): 
    return z**2 
    
def output(line): 
    print(line) 
    
r = add(1,2,3) 
q = subtract(r,1) 
output(f"{square(q)}")
'''
