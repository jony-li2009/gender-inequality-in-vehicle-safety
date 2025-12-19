import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt


fname = os.path.join("accident_fatal_data", "accident_fatal_2021.xlsx")
df1 = pd.read_excel(fname)

fname = os.path.join("accident_fatal_data", "accident_fatal_2020.xlsx")
df2 = pd.read_excel(fname)

fname = os.path.join("accident_fatal_data", "accident_fatal_2019.xlsx")
df3 = pd.read_excel(fname)

df = pd.concat([df1])
print(df.shape)
df = df[["FATAL1", "FATAL3","AGE1", "FEM1", "AGE3", "FEM3"]].dropna()
df = df[(df['AGE1'] >= 16) & (df['AGE1'] < 97) & (df['AGE3'] >= 16) & (df['AGE3'] < 97)]
print(df.shape)
#df = df[df['FATAL1'] == 2]
df['AGE1'] = df['AGE1']/100
df['AGE3'] = df['AGE3']/100
X = df[["AGE1", "FEM1", "AGE3", "FEM3"]].to_numpy()
y = np.array(df["FATAL1"].to_list())
print(X.shape)

# clf = RandomForestClassifier(max_depth=5, random_state=0, class_weight='balanced')
# clf.fit(X, y)
# print(clf.score(X, y))

#clf = AdaBoostClassifier(n_estimators=30, random_state=0).fit(X, y)
#print(clf.score(X, y))

#clf = MLPClassifier(hidden_layer_sizes=(50, ), random_state=1, max_iter=300).fit(X, y)
clf = LogisticRegression(random_state=1, max_iter=100, class_weight='balanced').fit(X, y)
print(clf.score(X, y))
print(clf.coef_, clf.intercept_)

age = np.arange(21, 97)/100
fem1 = np.ones(age.shape) * 0
age3 = np.ones(age.shape)
fem3 = np.ones(age.shape) * 0


fig = plt.figure()
ax = fig.add_subplot(111)
for m in age[::2]:
    data = np.array([age, fem1, age3 * m, fem3]).transpose()
    prob = clf.predict_proba(data)
    ax.plot(age * 100,  prob[:, 1], label=int(m * 100))

colormap = plt.cm.nipy_spectral #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1,len(ax.lines))]
for i,j in enumerate(ax.lines):
    j.set_color(colors[i])

ax.set_xticks(np.arange(0, 100, 10))
ax.set_yticks(np.arange(0, 1, 0.1))
ax.legend(loc=2, fontsize=5)
#ax.grid()
    
fig.savefig("passage.png")
# age3 = np.ones(age.shape) * 43/100
# data = np.array([age, fem1, age3, fem2]).transpose()
# prob2 = clf.predict_proba(data)

# plt.plot(age * 100,  prob1[:, 1], 'r', age*100, prob2[:, 1], 'g')
# plt.show()
#print(clf.score(X, y))
# prob = clf.predict_proba(X)
# Xs = df['AGE1'].to_list()
# Ys = prob[:, 1]
# plt.scatter(Xs, Ys)
# plt.show()

#clf = LogisticRegression(random_state=1, max_iter=100, class_weight='balanced').fit(X, y)

# clf_svc = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, class_weight='balanced'))
# clf_svc.fit(X, y)
# print(clf_svc.score(X, y))
# prob = clf_svc.predict_proba(X)
# Xs = df['AGE1'].to_list()
# Ys = prob[:, 1]
# plt.scatter(Xs, Ys)
# plt.show()


# print(clf.coef_, clf.intercept_)

# age = np.arange(-300, 500, 0.1)
# odd = clf.intercept_ + clf.coef_[0][0]* age + clf.coef_[0][1] * 1 + clf.coef_[0][2] * 40 + clf.coef_[0][3] * 1
# y = 1./(1+np.exp(-odd))
# plt.plot(age, y)
# plt.show()


# prob = clf.predict_proba(X)
#Xs = df['AGE1'].to_list()

#Ys = prob[:, 1]
#plt.scatter(Xs, Ys)
#plt.show()
#y = prob[:, 1]
#print(y[y < 0.5])
#print(sum(prob), prob.shape)
# y = np.array(df["FATAL3"].to_list())

# clf = LogisticRegression(random_state=1, max_iter=100).fit(X, y)

# print(clf.coef_, clf.intercept_)

# df_1 = df[["FATAL1", "AGE1", "FEM1", "AGE3", "FEM3"]]
# df_1["FATAL1"] = df_1["FATAL1"] - 1
# df_2 = df[["FATAL3", "AGE1", "FEM1", "AGE3", "FEM3"]]
# df_2["FATAL3"] = df_2["FATAL3"] - 1
# #ytrain = df[['FATAL1']] - 1

# log_reg = sm.logit("FATAL1~AGE1 + FEM1 + AGE3 + FEM3", data=df_1).fit()
# #print(log_reg.summary(df_1[["AGE1", "FEM1", "AGE3", "FEM3"]])) 

# age = np.array(range(21, 97))
# df = pd.DataFrame.from_dict({
#     "AGE1": age,
#     "FEM1": 0,
#     "AGE3": 30,
#     "FEM3": 1
# })

# y = log_reg.predict(df_1)
# print(np.sum(y < 0.5))

# age1 = np.array(range(-1000, 1000))
# coeff = log_reg.params.values
# odd = coeff[0] + coeff[1] * age1 + coeff[2] * 0 + coeff[3] * 35 + coeff[4] * 1
# y = 1./(1+np.exp(-odd))
# plt.plot(age1, y)
# plt.show()


#log_reg = sm.logit("FATAL3~AGE1 + FEM1 + AGE3 + FEM3", data=df_2).fit()
#print(log_reg.summary()) 

# ytrain = df[['FATAL3']] - 1 

# log_reg = sm.Logit(ytrain, Xtrain).fit()
# print(log_reg.summary()) 
