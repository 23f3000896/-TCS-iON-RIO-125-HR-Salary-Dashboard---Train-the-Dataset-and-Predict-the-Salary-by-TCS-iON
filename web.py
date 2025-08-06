'''from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")
    #return redirect(url_for("predict"))
@app.route("/predict",methods =["POST"])
def predict():
    Salary=[float(x)for x in request.form.values()]
    final_salary= [np.array(Salary)]
    output=model.predict(final_salary)
    return render_template("res.html",prediction_text="The Salary is Rs {}".format(output))

if __name__ =='__main__':
    app.run(debug=True)'''

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('HRDataset.csv')

# Choose relevant features, excluding EmpID
features = ['GenderID', 'DeptID', 'PerfScoreID']
# You can expand this list with features like 'EngagementSurvey', 'EmpSatisfaction', etc.

X = data[features]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(y_pred)

pickle.dump(regressor, open("model.pkl", "wb"))
