if request.method == 'POST':
        try:
            # Retrieve form data
            age = float(request.form['age'])
            salary = float(request.form['salary'])

            # Scale the input
            input_data = scaler.transform([[age, salary]])

            # Make prediction
            prediction = model.predict(input_data)
            result = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"

        except Exception as e:
            result = f"Error: {e}"