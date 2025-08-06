import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

Hr_data =pd.read_csv("HRDataset.csv")
print(Hr_data.head(5))
x= Hr_data[["EmpID","GenderID","DeptID","PerfScoreID"]]
y = Hr_data[["Salary"]]
Hr_data.drop(columns="Salary",inplace=True)
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=42, test_size=0.2)

from sklearn.linear_model import LogisticRegression

regressor= LogisticRegression()
regressor = regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
pickle.dump(regressor,open("model.pkl","wb"))