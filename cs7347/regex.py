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

match = re.search('r'\w\w\w', '@@abcd!!')
match.group()
