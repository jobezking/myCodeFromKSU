# Text: "purple alice-b@gmail.com monkey dishwasher"
# Use regex to find and extract the email address.
import re
str = "purple alice-b@gmail.com monkey dishwasher"
match = re.search(r'[\w\.-]+@[\w\.-]+', str)
if match:
    print('found email:', match.group())
else:
    print('did not find email')

# Group extraction feature of regular expression allows you to pick out parts of the matching tex
# () serves as logical group
str = "purple alice-b@gmail.com monkey dishwasher"
match = re.search('([\w.-]+)@([\w.-]+)', str)
if match:
    print(match.group()) # entire match alice-b@gmail.com
    print(match.group(1)) # username alice-b
    print(match.group(2)) # domain gmail.com    

# findall() finds all the matches and returns them as a list of strings
str = "purple alice@google.com, blah monkey bob@abc.com dishwasher"
emails = re.findall(r'[\w\.-]+@[\w\.-]+', str)
for email in emails:
    print(email)