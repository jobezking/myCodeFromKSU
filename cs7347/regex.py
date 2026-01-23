import re
str = "an example word:cat!!"
match = re.search(r'word:\w\w\w',str)
if match:
  print('found', match.group())
else:
  print('did not find')

### Search for pattern 'iii' in string 'piiig'.
### All of the pattern must match but it may appear anywhere.
### On success, match.group() is matched text.
match = re.search(r'iii', 'piiig')
match.group()

match = re.search(r'igs', 'piiig')
print(match)

## . = any char but \n
match = re.search('r..g', 'piiig')
match.group()

## \d = digit char, \w = word char
match = re.search(r'\d\d\d', 'p123g')
match.group()

match = re.search(r'\w\w\w', '@@abcd!!')
match.group()

## i+ = one or more i's, as many as possible
match = re.search(r'i+', 'piiig')
print('1.', match.group())
# OR
match = re.search(r'pi+', 'piiig')
print('1', match.group())
# Finds the first/leftmost solution and within it drives the +
# as far as possible. (aka 'leftmost and largest')
# In this example note that it does not get tot he second set of i's.
match = re.search(r'i+', 'piigiii')
print('2.', match.group())
# \s* = zero or more whitespace chars
# Here look for 3 digits, possibly separated by whitespace.
match = re.search(r'\d\s*\d\s*\d', 'xx1 2   3xx')
print('3.', match.group()) 
match = re.search(r'\d\s*\d\s*\d', 'xx12  3xx')
print('4.', match.group())
match = re.search(r'\d\s*\d\s*\d', 'xx123xx')
print('5.', match.group())
## ^ = matches the start of string so this fails:
match = re.search(r'^b\w+', 'foobar')
print('6.', match)
## but the ^ it succeeds:
match = re.search(r'^f\w+', 'foobar')
print('7.', match.group()) 

