//Takes a specific dropdown menu item by ID, a list of items and a placeholder text and fills dropdown with options
function populateSelect(selectId, listOfValues, placeholderText) {
    var selectElement = document.getElementById(selectId); // find dropdown element in html by ID

    // Safety Check: if element doesn't exist stop here and return error
    if (selectElement === null) {
        console.error("Could not find element with ID: " + selectId);
        return;
    }

    selectElement.innerHTML = "";     // clear out any existing options on page

    // create the default placeholder option 
    var defaultOption = document.createElement("option"); // Create <option></option>
    defaultOption.text = placeholderText;                 // Set the visible text
    defaultOption.value = "";                             // Set the value to empty
    defaultOption.disabled = true;                        // Make it un-clickable
    defaultOption.selected = true;                        // Make it selected by default

    selectElement.appendChild(defaultOption); // add default option to dropdown

    // loop through list of values and create option for each
    for (var i = 0; i < listOfValues.length; i++) {
        var value = listOfValues[i];

        var newOption = document.createElement("option");
        newOption.text = value;
        newOption.value = value;

        selectElement.appendChild(newOption);
    }
}

//wait for t"DOMContentLoaded" event to ensure the HTML is fully loaded before run script attempt
document.addEventListener("DOMContentLoaded", function () {

    // populate dropdowns for state, year, race, sex from LifeExpectancyPredictor.py through app.py
    populateSelect("state", states, "-- Select a State --");
    populateSelect("year", years, "-- Select a Year --");
    populateSelect("race", races, "-- Select a Race --");
    populateSelect("sex", sexes, "-- Select a Sex --");

    // need to handle dropdowns for state and county because county must exist in state 
    var stateDropdown = document.getElementById("state");
    var countyDropdown = document.getElementById("county");

    if (stateDropdown && countyDropdown) {     // both need to exist on page to proceed

        countyDropdown.disabled = true; // initially disable the county dropdown until state is selected

        // listen for state dropdown change event from page
        stateDropdown.addEventListener("change", function () { 
            var selectedState = stateDropdown.value; // get state value
            countyDropdown.innerHTML = "";  // clear current county list
            var countiesForThisState = stateCountyMap[selectedState]; // verify valid state and state-county pair

            if (selectedState !== "" && countiesForThisState !== undefined) {
                countyDropdown.disabled = false; // re-enable county dropdown
                // add placeholder 
                var defaultCountyOption = document.createElement("option");
                defaultCountyOption.text = "-- Select a County --";
                defaultCountyOption.value = "";
                defaultCountyOption.disabled = true;
                defaultCountyOption.selected = true;
                countyDropdown.appendChild(defaultCountyOption);
                // loop through all counties for state
                for (var j = 0; j < countiesForThisState.length; j++) {
                    var countyName = countiesForThisState[j];
                    var countyOption = document.createElement("option");
                    countyOption.text = countyName;
                    countyOption.value = countyName;
                    countyDropdown.appendChild(countyOption);
                }

            } else { // if user entered placeholder or invalid state lock county box again
                var resetOption = document.createElement("option");
                resetOption.text = "-- Select a State First --";
                resetOption.value = "";
                resetOption.disabled = true;
                resetOption.selected = true;
                countyDropdown.appendChild(resetOption);
                countyDropdown.disabled = true;
            }
        });
    }
    
    // see if app.py sent a prediction value to the page
    if (typeof prediction !== "undefined" && prediction !== null) {
        var predictionBox = document.getElementById("prediction-box");
        var predictionText = document.getElementById("prediction-text");
        predictionText.textContent = "Predicted Life Expectancy: " + prediction; // set text inside html element
        predictionBox.style.display = "block"; // change CSS 'display' property to make box visible
    }
});