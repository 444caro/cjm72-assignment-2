# Define your virtual environment and flask app
VENV = venv
FLASK_APP = app.py

# Install dependencies
install:
	python3 -m venv $(VENV)
	source $(VENV)/bin/pip install -r requirements.txt

# Run the Flask application
run:
	source $(VENV)/bin/activate && FLASK_APP=$(FLASK_APP) flask run --port 3000
# FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install
