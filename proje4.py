import pandas as pd
import seaborn as sns#Görselleştirme kütüphanesi
import numpy as np
import matplotlib.pyplot as plt#Görselleştirme kütüphanesi

from scipy import stats #İstatiksel kütüphaneler
from scipy.stats import  skew

from sklearn.preprocessing import RobustScaler #Makine Öğrenmesi Kütüphaneleri
from sklearn.linear_model import  Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error




# warning 
import warnings
warnings.filterwarnings('ignore')

column_name = ["MPG", "SilindirSayisi", "MotorHacmi","BeygirGucu","Agirlik","ivme","Yili", "Origin"]
data = pd.read_csv("auto-mpg.data", names = column_name, na_values = "?", comment = "\t",sep = " ", skipinitialspace = True)

data = data.rename(columns = {"MPG":"YakitTuketimi"})

describe = data.describe()

# %% EDA

corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()

threshold = 0.75 #Görüntü eşikleme değerimiz
filtre = np.abs(corr_matrix["YakitTuketimi"])>threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()


# %% 

thr = 2
BeygirGucu_desc = describe["BeygirGucu"]
q3_hp = BeygirGucu_desc[6]
q1_hp = BeygirGucu_desc[4]
IQR_hp = q3_hp - q1_hp
top_limit_hp = q3_hp + thr*IQR_hp
bottom_limit_hp = q1_hp - thr*IQR_hp
filter_hp_bottom = bottom_limit_hp < data["BeygirGucu"]
filter_hp_top = data["BeygirGucu"] < top_limit_hp
filter_hp = filter_hp_bottom & filter_hp_top

data = data[filter_hp]

ivme_desc = describe["ivme"]
q3_acc = ivme_desc[6]
q1_acc = ivme_desc[4]
IQR_acc = q3_acc - q1_acc # q3 - q1
top_limit_acc = q3_acc + thr*IQR_acc
bottom_limit_acc = q1_acc - thr*IQR_acc
filter_acc_bottom = bottom_limit_acc < data["ivme"]
filter_acc_top= data["ivme"] < top_limit_acc
filter_acc = filter_acc_bottom & filter_acc_top

data = data[filter_acc] # remove BeygirGucu outliers

# %% Feature Engineering
# Skewness

# YakitTuketimi dependent variable

data["YakitTuketimi"] = np.log1p(data["YakitTuketimi"]) 


# qq plot
plt.figure()
stats.probplot(data["YakitTuketimi"], plot = plt)
plt.show()

# feature - independent variable 
skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame(skewed_feats, columns = ["skewed"])

"""
Box Cox Transformation
"""
# %% one hot encoding
data["SilindirSayisi"] = data["SilindirSayisi"].astype(str)  
data["Origin"] = data["Origin"].astype(str) 

data = pd.get_dummies(data)

# %% Split - Stand 

# Split
x = data.drop(["YakitTuketimi"], axis = 1)
y = data.YakitTuketimi

test_size = 0.9
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = test_size, random_state = 42)

# Standardization
scaler = RobustScaler()  # RobustScaler #StandardScaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Lasso Regression (L1)

lasso = Lasso(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error',refit=True)
clf.fit(X_train,Y_train)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']


lasso = clf.best_estimator_
print("Lasso en iyi Alpha değeri: ",lasso)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Lasso Hata değeri: ",mse)
print("---------------------------------------------------------------")

plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha değeri")
plt.ylabel("sonuç")
plt.title("Lasso Hata değeri")



"""
StandardScaler:
    Linear Regression (hata değeri):  0.020
    Ridge (hata değeri):  0.019
    Lasso (hata değeri):  0.017
    ElasticNet (hata değeri):  0.017
    XGBRegressor (hata değeri): 0.017
    Averaged Models (hata değeri): 0.016
RobustScaler:
    Linear Regression (hata değeri):  0.020
    Ridge (hata değeri):  0.018
    Lasso (hata değeri):  0.016
    ElasticNet (hata değeri):  0.017
    XGBRegressor (hata değeri): 0.017
    Averaged Models (hata değeri): 0.015
"""