import pandas as pd
import numpy as np
from numpy import mean
import matplotlib as mp
import matplotlib.pyplot as plt
import os
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score, plot_confusion_matrix
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_validate, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDRegressor, SGDClassifier #LogisticRegression is a classification model
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import seaborn as sns
import numpy as np
from imageio import imread
import pandas as pd
from time import time as timer
import statsmodels.api as sm
import pylab as py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import tensorflow as tf
import plotly.graph_objects as go


from PIL import Image
from matplotlib import animation
from matplotlib import cm
from IPython.display import HTML
#Load dataset

path=r"D:\Romain\Documents\Biberach PhD\projet\Formations Bern\CAS ADS\M3\projet\Cell viability and extrusion dataset V1.csv"
bioprint_df = pd.read_csv(path) 

#Remove reference and DOI
bioprint_df = bioprint_df.drop(['Reference'], axis = 1)
bioprint_df = bioprint_df.drop(['DOI'], axis = 1)
bioprint_df = bioprint_df.drop(['Extrusion_Rate_Volume-wise_(mL/s)'], axis = 1)

print(bioprint_df.head())
print(bioprint_df.shape)
print(bioprint_df.isna().sum())

# #Imputting mode temperatures


imputer_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent') #imputing mode value into missing values for temperatures
bioprint_df.loc[:,['Syringe_Temperature_(°C)','Substrate_Temperature_(°C)',"Final_PEGMA_Conc_(%w/v)","Nozzle_Movement_Speed_(mm/s)","Fiber_Spacing_(µm)","Cell_Density_(cells/mL)"]] = imputer_mode.fit_transform(bioprint_df.loc[:,['Syringe_Temperature_(°C)','Substrate_Temperature_(°C)',"Final_PEGMA_Conc_(%w/v)","Nozzle_Movement_Speed_(mm/s)","Fiber_Spacing_(µm)","Cell_Density_(cells/mL)"]])



#Drop certain material concentration as no concentration values exist in papers
bioprint_df = bioprint_df.drop(['Fiber_Diameter_(µm)'], axis = 1)
#drop for extrusion pressure dataset creation
bioprint_df = bioprint_df.drop(['CaCl2_Conc_(mM)','NaCl2_Conc_(mM)','BaCl2_Conc_(mM)','SrCl2_Conc_(mM)','Physical_Crosslinking_Durantion_(s)','Photocrosslinking_Duration_(s)'], axis = 1) 
# #drop these variables to create the extrusion pressure dataset from the cell viability dataset

bioprint_df = bioprint_df.dropna(axis = 1, thresh=177)


#Drop redundant variables for Mondal Intrastudy dataset creation
print(bioprint_df.columns)
bioprint_df=bioprint_df.drop(['Cell_Culture_Medium_Used?','DI_Water_Used?','Precrosslinking_Solution_Used?','Saline_Solution_Used?','EtOH_Solution_Used?','Photoinitiator_Used?','Enzymatic_Crosslinker_Used?','Matrigel_Used?','Conical_or_Straight_Nozzle','Primary/Not_Primary'], axis = 1)
print(bioprint_df.columns)

#Drop instances without cell viability values
bioprint_df = bioprint_df[bioprint_df['Viability_at_time_of_observation_(%)'].notna()]



#Drop nonprinting instances (instances were extrusion pressure is zero)
bioprint_df = bioprint_df.drop(bioprint_df[bioprint_df['Extrusion_Pressure (kPa)'] == 0 ].index)
bioprint_df = bioprint_df[bioprint_df['Extrusion_Pressure (kPa)'].notna()] 
#used to create extrusion pressure dataset


#Feature Selection Through Correlation
corr = bioprint_df.corr()
print(corr)
fig, ax = plt.subplots(figsize = (20, 16))
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, 
linewidths=0.1)
print(abs(bioprint_df.corr()["Viability_at_time_of_observation_(%)"]))
# plt.close()


#Drop not correlated columns

bioprint_df = bioprint_df.drop(['Final_MeHA_Conc_(%w/v)','Final_NorHA_Conc_(%w/v)','Final_Chitosan_Conc_(%w/v)','Final_CS-AEMA_Conc_(%w/v)','Final_TCP_Conc_(%w/v)'], axis = 1)
bioprint_df = bioprint_df.drop(['Final_PEGTA_Conc_(%w/v)','Final_PEGMA_Conc_(%w/v)','Final_PEGDA_Conc_(%w/v)'],axis=1)


bioprint_df = bioprint_df.drop(['Acceptable_Viability_(Yes/No)'],axis=1)
bioprint_df = bioprint_df.drop(['Acceptable_Pressure_(Yes/No)'],axis=1)


# ## Imputing Values
# #Imputation of numerical/continuous values databases

imputer_knn = KNNImputer(n_neighbors = 30, weights = "uniform") #imputing mode value into missing values
bioprint_df.iloc[:,0:28] =  imputer_knn.fit_transform(bioprint_df.iloc[:,0:28]) #used for cell  viability dataset preprocessing
print(bioprint_df.head())
print(bioprint_df.shape)
print(bioprint_df.isna().sum())


##Normalizing/Scalarizing and Encoding Continuous and  Categorical Data
x = bioprint_df.drop(["Viability_at_time_of_observation_(%)"],axis=1)

print(f"x: {x}")
y = bioprint_df["Viability_at_time_of_observation_(%)"].values
print(f"y: {y}")

#Distribution normal or not
for i in range(len(bioprint_df.columns)):

    plt.figure(figsize=(10,10))

    plt.hist(bioprint_df.iloc[:,i].values, bins=50)

    plt.gca().set(title='Histogramm', ylabel='Frequency',xlabel=bioprint_df.columns[i]);

    plt.legend()
    plt.show()
    sm.qqplot(bioprint_df.iloc[:,i], line ='45')
    py.show()

# D'agostino-Pearson test
for i in range(len(bioprint_df.columns)):
    k2, p = stats.normaltest(bioprint_df.iloc[:,i]) # D Agostino-Pearson. The method returns the test statistic value and the p-value
    alpha = 0.001 # Rejection criterion defined by you
    print('Alpha = ',alpha)
    print('p = ',p)
    if p < alpha:  # null hypothesis: x comes from a normal distribution
         print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")
        
#plots
plt.figure(figsize=(10,10))

plt.scatter(bioprint_df["Substrate_Temperature_(°C)"],bioprint_df["Viability_at_time_of_observation_(%)"],c='b')
plt.show()
for i in range(len(bioprint_df.columns)-1):
    plt.figure(figsize=(10,10))

    plt.scatter(bioprint_df.iloc[:,i],bioprint_df["Viability_at_time_of_observation_(%)"],color='blue')
    plt.xlabel(bioprint_df.columns[i])
    plt.ylabel("Viability_at_time_of_observation_(%)")
    plt.show()

##Machine Learning Algorithms for Regression Modeling
#Linear Regression
import statsmodels.api as sm
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1, random_state = 42)
lr = LinearRegression()
lr.fit(x_train,y_train)
pred_lr = lr.predict(x_test) #runs label prediction on the test set
lr_score = lr.score(x_test,y_test) #returns the coefficient of determination of the model
print(f" rsquare= {lr_score}") #prints the coefficient of determination of the model
print('train R2 =', lr.score(x_train, y_train))
print('test R2 =', lr.score(x_test, y_test))

print('train mse =', np.std(y_train - lr.predict(x_train)))
print('test mse =', np.std(y_test - lr.predict(x_test)))

#OLS regression


x = sm.add_constant(x)
lin_model = sm.OLS(y, x)
regr_results = lin_model.fit()
print(regr_results.summary()) 
plt.text(0.01, 0.05, str(regr_results.summary()), {'fontsize': 10}, fontproperties = 'monospace')
plt.show()



#Polynomial regression

from sklearn.preprocessing import PolynomialFeatures
x=x.values
#define our polynomial model, with whatever degree we want
degree=2

# PolynomialFeatures will create a new matrix consisting of all polynomial combinations 
# of the features with a degree less than or equal to the degree we just gave the model
poly_model = PolynomialFeatures(degree=degree)

# transform out polynomial features
poly_x_values= poly_model.fit_transform(x)

# should be in the form [1, a, b, a^2, ab, b^2]
print(f'initial values {x[0]}\nMapped to {poly_x_values[0]}')


# let's fit the model
poly_model.fit(poly_x_values, y)

# we use linear regression as a base!!! ** sometimes misunderstood **
regression_model = LinearRegression()

regression_model.fit(poly_x_values, y)

y_pred = regression_model.predict(poly_x_values)
print(y_pred)

regression_model.coef_

mean_squared_error(y, y_pred, squared=False)







X, y = bioprint_df.drop(["Viability_at_time_of_observation_(%)"],axis=1), bioprint_df["Viability_at_time_of_observation_(%)"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=42)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)


poly_reg_y_predicted = poly_reg_model.predict(X_test)
from sklearn.metrics import mean_squared_error
poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
print(poly_reg_rmse)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
lin_reg_y_predicted = lin_reg_model.predict(X_test)
lin_reg_rmse = np.sqrt(mean_squared_error(y_test, lin_reg_y_predicted))
print(lin_reg_rmse)



for i in range(1,5):

    X, y = bioprint_df.drop(["Viability_at_time_of_observation_(%)"],axis=1), bioprint_df["Viability_at_time_of_observation_(%)"]
    poly = PolynomialFeatures(degree=i, include_bias=False)
    poly_features = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=42)

    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_train, y_train)


    poly_reg_y_predicted = poly_reg_model.predict(X_test)
    from sklearn.metrics import mean_squared_error
    poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
    print(poly_reg_rmse)




#Random Forest Regressor
x = bioprint_df.drop(["Viability_at_time_of_observation_(%)"],axis=1)
y = bioprint_df["Viability_at_time_of_observation_(%)"].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1, random_state = 42)
rfr = RandomForestRegressor(max_depth=6,random_state = 42, n_estimators=10) 
rfr.fit(x_train,y_train)
pred_rfr = rfr.predict(x_test) #runs label prediction on the test set
rfr_score = rfr.score(x_test, y_test) #returns the coefficient of determination of the model


# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, pred_rfr)), '.3f'))
print("\nRMSE: ", rmse)

print(f"rfr score: {rfr_score}") #coefficient of determination scoring

# Used to create random forest based feature importance ranking graph
features = x_train.columns
importances = rfr.feature_importances_
indices = np.argsort(importances)
# customized number of the most important features
num_features = 10
#plt.figure(figsize=(10,100))
#plt.title('Random Forest Regression Feature Importances')
# only plot the customized number of features
#Plots a bar graph of the relative feature importance values of the most importance features
plt.barh(range(num_features), importances[indices[-num_features:]], 
color='b', align='center')
plt.yticks(range(num_features), [features[i] for i in indices[-
num_features:]])
plt.xlabel('Relative Importance')
plt.xlim(0,0.6)
plt.show()




# Limit depth of tree to 3 levels
from sklearn.tree import export_graphviz
import pydot
rfr = RandomForestRegressor(max_depth=6,random_state = 42, n_estimators=10) 
rfr.fit(x_train,y_train)
feature_list = list(bioprint_df.drop(["Viability_at_time_of_observation_(%)"],axis=1).columns)
print(len(feature_list ))


# #Support vector regression

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size 
= 0.1, random_state = 42)
svr = SVR(kernel = 'rbf')
svr.fit(x_train,y_train)
pred_svr = svr.predict(x_test) #runs label prediction on the test set
svr_score = svr.score(x_test,y_test) #returns the coefficient of determination of the model

#aur = roc_auc_score(y_test,pred_svr)
mae = mean_absolute_error(y_test,pred_svr)
mse = mean_squared_error(y_test,pred_svr)
print(mae)
print(mse)
print(f"svr score: {svr_score}") #prints the coefficient of determination of the model
# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, pred_svr)), '.3f'))
print("\nRMSE: ", rmse)


