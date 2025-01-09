import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'

df = pd.read_csv(filepath)
print(df.head(5))

#Display the data types of each column using the function dtypes. 
print(df.dtypes)

#We use the method describe to obtain a statistical summary of the dataframe.
print(df.describe())

#Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), 
#then use the method describe() to obtain a statistical summary of the data

df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
df = df.iloc[:, 1:]
print(df.describe())

#We can see we have missing values for the columns  bedrooms and  bathrooms 
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

#replace the missing values of the column 'bedrooms' with the mean of the column 'bedrooms'  
#using the method replace(). Don't forget to set the inplace parameter to True

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
#We also replace the missing values of the column 'bathrooms' with the mean of the column 'bathrooms'  using the method replace(). 
#Don't forget to set the  inplace  parameter top  True 
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

#Module 3: Exploratory Data Analysis

#Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a data frame. 
#Take a screenshot of your code and output. 
# You will need to submit the screenshot for the final project.
print(df["floors"].value_counts())
floor_counts = df["floors"].value_counts()
floor_counts_df = floor_counts.to_frame()
floor_counts_df.rename(columns={"floors": "count"}, inplace=True)
print(floor_counts_df)

#Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more 
#price outliers. Take a screenshot of your code and boxplot. You will need to submit the screenshot for the final project.

sns.boxplot(x="waterfront", y="price", data=df)
plt.title("Distribución de Precios por Vista al Agua (Waterfront)", fontsize=14)
plt.xlabel("Vista al Agua (0 = No, 1 = Sí)", fontsize=12)
plt.ylabel("Precio (USD)", fontsize=12)
plt.show()

#Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price. 
#Take a screenshot of your code and scatterplot. You will need to submit the screenshot for the final project.

sns.regplot(x="sqft_above", y="price", data=df, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
plt.title("Relación entre sqft_above y price", fontsize=14)
plt.xlabel("Superficie sobre el suelo (sqft_above)", fontsize=12)
plt.ylabel("Precio (USD)", fontsize=12)
plt.show()

#We can use the Pandas method corr() to find the feature other than price that is most correlated with price.
print(df.corr()['price'].sort_values())

#Model Development
#We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)
print(lm.score(X, Y))

#Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2. 
#Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
X = df[['sqft_living']]
Y = df['price']
lm1 = LinearRegression()
lm1.fit(X,Y)
lm1.score(X, Y)
print(lm1.score(X, Y))

#Fit a linear regression model to predict the 'price' using the list of features:

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     

#Then calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
lm2 = LinearRegression()
X = df[features]
Y = df['price']
lm2.fit(X,Y)

# Calcular el R^2
r2_score = lm2.score(X, Y)

# Mostrar el R^2
print("R^2:", r2_score)

#This will help with Question
#Create a list of tuples, the first element in the tuple contains the name of the estimator:

#'scale'

#'polynomial'

#'model'

#The second element in the tuple contains the model constructor

#StandardScaler()

#PolynomialFeatures(include_bias=False)

#LinearRegression()

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

#Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list features, and calculate the R^2. 
#Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
from sklearn.metrics import r2_score
pipe=Pipeline(Input)

X = X.astype(float)
pipe.fit(X,Y)
ypipe=pipe.predict(X)
#print(Y,ypipe) #casas por indice y valores de las casas
r2 = r2_score(Y, ypipe)
print("R^2:", r2)

#Model Evaluation and Refinement
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")

#We will split the data into training and testing sets:

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

#Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.
#Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
from sklearn.linear_model import Ridge
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))

#Perform a second order polynomial transform on both the training data and testing data. 
#Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, 
#and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2. You will need to submit it for the final project.
pr = PolynomialFeatures(degree=2)

x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))


