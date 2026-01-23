#Shakespeare's Twelfth Night
#Using Beautiful Soup
from bs4 import BeautifulSoup
from urllib import urlopen

def scrapeHTML(url):
    page = urlopen(url).read()
    soup = BeautifulSoup(page)
    return soup.p.contents

if __name__ == "__main__":
    url = "http:bit.ly/1D7wKcH"
    text = scrapeHTML(url)
    print(text)