import os
from typing import List
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import xgboost as xgb
import matplotlib.pyplot as plt
import openpyxl

def encode_cycle(x, max_value):
    sinx = np.sin(2 * np.pi * x/max_value)
    cosx = np.cos(2 * np.pi * x/max_value)
    return sinx, cosx


def adjust_sheet_column_width(sheet, min_width:int=10, max_width:int=70):
    """_summary_

    Args:
        sheet (_type_): _description_
        min_width (int, optional): _description_. Defaults to 5.
        max_width (int, optional): _description_. Defaults to 50.
    """
    dims = {}
    for row in sheet.rows:
        for cell in row:
            if cell.value:
                dims[cell.column_letter] = max((dims.get(cell.column_letter, 0), len(str(cell.value))))

    for col, value in dims.items():
        sheet.column_dimensions[col].width = min(max(value, min_width), max_width)

def save_multiple_dfs(df_list: List[pd.DataFrame], sheet_names: List[str], file_name:str):
    """_summary_

    Args:
        df_list (List[pd.DataFrame]): _description_
        sheet_names (List[str]): _description_
        file_name (str): _description_
    """
    writer = pd.ExcelWriter(file_name,engine='openpyxl')
    for df, name in zip(df_list, sheet_names):
        df.to_excel(writer, sheet_name=name, index=False)   
    writer.close()

    wb = openpyxl.load_workbook(file_name)
    for name in wb.sheetnames:
        sheet = wb[name]
        adjust_sheet_column_width(sheet)
        
    wb.save(filename=file_name)
    
    
def visualize_random_forest(clf, feature_names, target_names, filename):
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    tree.plot_tree(clf.estimators_[0],
               feature_names = feature_names, 
               class_names=target_names,
               filled = True);
    fig.savefig(filename)
    

class FARSModel():
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.min_age = 21
        self.max_age = 97
        self.class_weight = "balanced"
        
        self.person_vars = ["AGE1", "FEM1", "AGE3", "FEM3"]
        self.env_vars = ["MONTH", "DAY", "HOUR", "WEATHER", "LGT_COND"]
        self.type = "RandomForest" #"LogisticRegression"
        
    def _get_data(self):
        dfs =list()
        for i in self.years:
            fname = os.path.join(self.data_path, f"accident_{i}.xlsx")
            df = pd.read_excel(fname)
            df = df.dropna()
            df = df[(df["AGE1"] >= self.min_age) & (df["AGE1"] < self.max_age) & (df["AGE3"] >= self.min_age) & (df["AGE3"] < self.max_age)].reset_index(drop=True)
            if self.is_fatality:
                #only consider the data with at least one fatality
                inj_sev = 4
            else:
                #only consider the data with at least one major injury
                inj_sev = 3
                
            df = df[(df["FATAL1"] <= inj_sev) & (df["FATAL3"] <= inj_sev)].reset_index(drop=True)
            df = df[(df["FATAL1"] == inj_sev) | (df["FATAL3"] == inj_sev)].reset_index(drop=True)
            df.loc[df["FATAL1"] != inj_sev, "FATAL1"] = 2
            df.loc[df["FATAL1"] == inj_sev, "FATAL1"] = 1
            df.loc[df["FATAL3"] != inj_sev, "FATAL3"] = 2
            df.loc[df["FATAL3"] == inj_sev, "FATAL3"] = 1
            
            dfs.append(df)
            
        return pd.concat(dfs)
    
    def fit_model(self):
        X = self.df_data[self.person_vars].to_numpy()
        y1 = self.df_data["FATAL1"].to_list()
        y3 = self.df_data["FATAL3"].to_list()
        if self.type == "RandomForest":
            clf_driver = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0).fit(X, y1)
            clf_passenger = RandomForestClassifier(n_estimators=100,  max_depth=6, random_state=0).fit(X, y3)
        elif self.type == "XGBoost":
            pass
        else:
            clf_driver = LogisticRegression(random_state=0, max_iter=100, class_weight=self.class_weight).fit(X, y1)
            clf_passenger = LogisticRegression(random_state=0, max_iter=100, class_weight=self.class_weight).fit(X, y3)
            
        score1 = clf_driver.score(X, y1)
        print(score1)
        score3 = clf_passenger.score(X, y3)
        print(score3)
        
        return clf_driver, clf_passenger
    
    def _calc_age(self, X, clf1, clf2):
        risk = list()
        for age in range(self.min_age, self.max_age):
            X[:, 0] = age
            pred1 = clf1.predict_proba(X)
            pred2 = clf2.predict_proba(X)
            #value = sum([a[0]/b[0]for a, b in zip(pred1, pred2)])/len(pred1)
            value = sum(pred1[:, 0])/sum(pred2[:, 0])
            risk.append(value)
            
        return risk
    
    def _calc_average_risk(self, X, idx, clf1, clf2):
        #Male driver with different age
        X1 = np.array(X)
        X1[:, idx] = 0
        pred1 = clf1.predict_proba(X1)
        pred2 = clf2.predict_proba(X1)
        #male_risk = sum([a[0]/b[0]for a, b in zip(pred1, pred2)])/len(pred1)
        male_risk = sum(pred1[:, 0])/sum(pred2[:, 0])
        
        X1[:, idx] = 1
        pred1 = clf1.predict_proba(X1)
        pred2 = clf2.predict_proba(X1)
        #female_risk = sum([a[0]/b[0]for a, b in zip(pred1, pred2)])/len(pred1)
        female_risk = sum(pred1[:, 0])/sum(pred2[:, 0])
        
        return male_risk, female_risk
    
    def _calc_model_year(self, clf_driver, clf_passenger, start_year=1995):
        model_years = list()
        driver_male_risk = list()
        driver_female_risk = list()
        passenger_male_risk = list()
        passenger_female_risk = list()
        df = self.df_data[self.df_data['Model Year'] < start_year].reset_index(drop=True)
        if not df.empty:
            X = df[self.person_vars].to_numpy()
            risk1 = self._calc_average_risk(X, 1, clf_driver, clf_passenger)
            driver_male_risk.append(risk1[0])
            driver_female_risk.append(risk1[1])
            risk2 = self._calc_average_risk(X, 3, clf_passenger, clf_driver)
            passenger_male_risk.append(risk2[0])
            passenger_female_risk.append(risk2[1])
            model_years.append(f"<{start_year}")
            
        for year in range(start_year, 2022):
            df = self.df_data[self.df_data['Model Year'] == year].reset_index(drop=True)
            if not df.empty:
                X = df[self.person_vars].to_numpy()
                risk1 = self._calc_average_risk(X, 1, clf_driver, clf_passenger)
                driver_male_risk.append(risk1[0])
                driver_female_risk.append(risk1[1])
                risk2 = self._calc_average_risk(X, 3, clf_passenger, clf_driver)
                passenger_male_risk.append(risk2[0])
                passenger_female_risk.append(risk2[1])
                model_years.append(f"{year}")
                
        #2010
        df = self.df_data[self.df_data['Model Year'] < 2010].reset_index(drop=True)
        X = df[self.person_vars].to_numpy()
        driver_male1,driver_female1  = self._calc_average_risk(X, 1, clf_driver, clf_passenger)
        passenger_male1,  passenger_female1= self._calc_average_risk(X, 3, clf_passenger, clf_driver)
        print(driver_male1, passenger_male1, driver_female1, passenger_female1)
        
        df = self.df_data[self.df_data['Model Year'] > 2010].reset_index(drop=True)
        X = df[self.person_vars].to_numpy()
        driver_male2,driver_female2  = self._calc_average_risk(X, 1, clf_driver, clf_passenger)
        passenger_male2,  passenger_female2= self._calc_average_risk(X, 3, clf_passenger, clf_driver)
        print(driver_male2, passenger_male2, driver_female2, passenger_female2)
        
        df = pd.DataFrame.from_dict({
            "Model Year": model_years,
            "Male Driver": driver_male_risk,
            "Female Driver": driver_female_risk,
            "Male Passenger": passenger_male_risk,
            "Female Passenger": passenger_female_risk
        })
        
        return df
        
    
    def calc_risk(self, clf_driver, clf_passenger):
        X = self.df_data[self.person_vars].to_numpy()
        #Male driver with different age
        X_M = np.array(X)
        X_M[:, 1] = 0
        male_driver_risk = self._calc_age(X_M, clf_driver, clf_passenger)            
        #Female driver with different age
        X_F = np.array(X)
        X_F[:, 1] = 1
        female_driver_risk = self._calc_age(X_F, clf_driver, clf_passenger)
        #Male Passenger with different age
        X_M = np.array(X)
        X_M[:, 3] = 0
        male_passenger_risk = self._calc_age(X_M, clf_passenger, clf_driver)
        #Female Passenger with different age
        X_F = np.array(X)
        X_F[:, 3] = 1
        female_passenger_risk = self._calc_age(X_F, clf_passenger, clf_driver)
            
        df_age = pd.DataFrame.from_dict({
            "Age": list(range(self.min_age, self.max_age)),
            "Male Driver": male_driver_risk,
            "Female Driver": female_driver_risk,
            "Male Passenger": male_passenger_risk,
            "Female Passenger": female_passenger_risk
        })
        
        df_model_year = self._calc_model_year(clf_driver, clf_passenger)
        
        return df_age, df_model_year
        
    
    def run(self,
            years: List[list],
            is_fatality: bool,
            person_only: bool):
        self.years = years
        self.person_only = person_only
        self.is_fatality = is_fatality
        self.df_data = self._get_data()
        
        clf_driver, clf_passenger = self.fit_model()
        feature_names = self.person_vars
        target_names = ["YES", 'NO']
        if self.type == "RandomForest":
            print(clf_driver.feature_importances_)
            print(clf_passenger.feature_importances_)
            visualize_random_forest(clf_driver, feature_names, target_names, "driver_RF_model.png")
            visualize_random_forest(clf_passenger, feature_names, target_names, "passenger_RF_model.png")
        else:
            print(clf_driver.coef_, clf_driver.intercept_)
            print(clf_passenger.coef_, clf_passenger.intercept_)
        X = self.df_data[self.person_vars].to_numpy()
        risk1 = self._calc_average_risk(X, 1, clf_driver, clf_passenger)
        risk2 = self._calc_average_risk(X, 3, clf_passenger, clf_driver)
        print(risk1, risk2, (risk1[1] + risk2[1])/(risk1[0] + risk2[0]))
        
        df_age, df_model_year = self.calc_risk(clf_driver, clf_passenger)
        
        male_env_pro = {name: defaultdict(int) for name in self.env_vars}
        female_env_pro = {name: defaultdict(int) for name in self.env_vars}
        
        for i, row in self.df_data.iterrows():
            if row['FATAL1'] == 1:
                if row["FEM1"] == 0:
                    for name in self.env_vars:
                        male_env_pro[name][int(row[name])] += 1
                else:
                    for name in self.env_vars:
                        female_env_pro[name][int(row[name])] += 1
                        
            if row['FATAL3'] == 1:
                if row["FEM3"] == 0:
                    for name in self.env_vars:
                        male_env_pro[name][int(row[name])] += 1
                else:
                    for name in self.env_vars:
                        female_env_pro[name][int(row[name])] += 1
        
        df_envs = list()
        for i, name in enumerate(self.env_vars):
            items = list(male_env_pro[name].keys()) + list(female_env_pro[name].keys())
            items = sorted(list(set(items)))
            
            male_values = [male_env_pro[name][v]for v in items]
            total = sum(male_values)
            male_values = [v/total for v in male_values]
    
            female_values = [female_env_pro[name][v]for v in items]
            total = sum(female_values)
            female_values = [v/total for v in female_values]
            
            df = pd.DataFrame.from_dict({
                name: items,
                "Male": male_values,
                "Female": female_values,
                "": None
            })
            df_envs.append(df)
        
        df_env = pd.concat(df_envs, axis=1)
        save_multiple_dfs([df_age, df_model_year, df_env], ["age", "model year", "time-weather"], f"severe_injury_result_sum_{self.type}.xlsx")
         

data_path = "accident_data_full"
years  = list(range(2015, 2022))

FARS = FARSModel(data_path)
FARS.run(years, False, False)
