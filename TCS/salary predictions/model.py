import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ✅ Load your dataset (use relative path)
data = pd.read_csv(r"C:\Users\user\Desktop\TCS\salary prediction\HRDataset.csv")


# ✅ Select features and target column
features = ['GenderID', 'DeptID', 'PerfScoreID']
X = data[features]
y = data['Salary']

# ✅ Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

pickle.dump(regressor, open("model.pkl", "wb"))

print("✅ Model trained and saved successfully as model.pkl!")
