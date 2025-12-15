Life Expectancy Calculator 

sensible\_browser presumes Linux desktop, otherwise copy and paste URL into browser.
Note: "pip install \-r requirements.txt" only needs to be run once.

To run as Python program:  
pip install \-r requirements.txt  
python app.py  
sensible-browser http://localhost:5000  
When done:  
Press CTRL+C to quit

To run as flask application:  
pip install \-r requirements.txt  
flask run  
sensible-browser http://localhost:8888  
When done:  
Press CTRL+C to quit 

To run in Docker (Docker commands may require sudo):   
docker build \-t life-expectancy-app .  
docker run \-p 8888:8888 life-expectancy-app &  
sensible-browser http://localhost:8888  
When done:  
docker ps   
\# obtain the value from the "NAMES" column i.e. interesting\_less  
docker kill interesting\_less

To run in Docker Compose (Docker commands may require sudo):  
docker compose up \--build \-d  
sensible-browser http://localhost:8888  
When done:  
docker compose down
