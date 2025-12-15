from flask import Flask, render_template, request
from ensemble import LifeExpectancyPredictorEngine
import os

# --- Flask App Setup ---
# Point 'templates' to the 'templates' folder.
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)


# --- LOAD THE MODEL (ONCE) ---
# This is the key: We create the predictor object ONE TIME when the app starts.
# The model will be loaded and trained now and kept in memory.
# All users who visit the site will use this single, ready-to-go object.
print("Flask app starting... Initializing model.")
try:
    predictor = LifeExpectancyPredictorEngine()
    
    # Get the dropdown options from the engine (also only once)
    dropdown_options = predictor.get_dropdown_options()
    
    print("Flask app is trained and ready to serve predictions.")
    
except Exception as e:
    print(f"FATAL ERROR: Could not initialize model. {e}")
    # If the model fails to load, we can't run the app.
    predictor = None
    dropdown_options = {
        'states': [], 'years': [], 'races': [], 'sexes': [], 'state_county_map': {}
    }


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Main route that handles both displaying the form (GET)
    and processing the prediction (POST).
    """
    
    # If the model failed to load, show an error page.
    if predictor is None:
        return "Error: Model could not be loaded. Please check server logs.", 500
        
    prediction_result = None

    if request.method == 'POST':
        try:
            # --- Get data from the form ---
            state = request.form.get('state')
            county = request.form.get('county')
            # Year needs to be an integer
            year = int(request.form.get('year'))
            race = request.form.get('race')
            sex = request.form.get('sex')

            # --- Make the prediction ---
            # We use the 'predictor' object that's already in memory
            raw_prediction = predictor.make_prediction(
                state=state,
                county=county,
                year=year,
                race=race,
                sex=sex
            )
            
            # Format the result for display
            prediction_result = f"{raw_prediction:.2f} years"

        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction_result = f"Error: Could not make prediction. {e}"

    # --- Render the page ---
    # This will run for both GET and POST requests.
    # For a POST, it re-renders the page with the prediction_result.
    # For a GET, prediction_result is None, so the 'if' block in HTML is false.
    return render_template(
        'index.html',
        prediction=prediction_result,
        states=dropdown_options['states'],
        years=dropdown_options['years'],
        races=dropdown_options['races'],
        sexes=dropdown_options['sexes'],
        # Pass the map to the template. The 'tojson' filter in HTML will handle it.
        state_county_map=dropdown_options['state_county_map'] 
    )

if __name__ == '__main__':
    # Run the Flask app
    # Set debug=False for a production environment
    app.run(debug=True)