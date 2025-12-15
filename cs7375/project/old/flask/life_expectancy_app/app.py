from flask import Flask, request, render_template
from life_expectancy_predictor import predict_life_expectancy

app = Flask(__name__)

# Predefined dropdown options
STATES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","District of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois",
    "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts",
    "Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota",
    "Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina",
    "South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington",
    "West Virginia","Wisconsin","Wyoming"
]

RACES = ["All Races", "White", "Black"]
SEXES = ["Both Sexes", "Male", "Female"]
YEARS = list(range(1900, 2019))  # 1900–2018 inclusive

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        state = request.form["state"]
        county = request.form["county"]
        year = int(request.form["year"])
        race = request.form["race"]
        sex = request.form["sex"]

        # Now only receive a single float estimate from the predictor
        final = predict_life_expectancy(state, county, year, race, sex) # returns Numpy array of one element
        prediction = f"{final[0]:.2f}"

    return render_template(
        "index.html",
        prediction=prediction,
        states=STATES,
        races=RACES,
        sexes=SEXES,
        years=YEARS
    )

if __name__ == "__main__":
    app.run(debug=True)
