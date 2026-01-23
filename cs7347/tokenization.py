import random
import re

text = "The minimum edit distance algorithm, an \
    example of the class of dynamic programming algorithms."
words = re.findall(r'\w+', text)
print(words)