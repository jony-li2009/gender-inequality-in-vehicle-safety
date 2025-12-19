import os
from typing import List
import json
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
import openpyxl

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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



def model_fit(X, y):
    #clf = LogisticRegression(random_state=1, max_iter=100, class_weight='balanced').fit(X, y)
    clf = MLPClassifier(hidden_layer_sizes=(50, ), random_state=1, max_iter=300).fit(X, y)
    score = clf.score(X, y)
    return clf, score


class FARS_Model():
    def __init__(self,
                 data_path: str,
                 data_years: List,
                 prefix: str="accident_fatal_",
                 min_age: int=16,
                 max_age: int=97) -> None:
        self.min_age = min_age
        self.max_age = max_age
        dfs = list()
        for year in data_years:
            fname = os.path.join(data_path, f"{prefix}{year}.xlsx")
            df = pd.read_excel(fname)
            df = df[["FATAL1", "FATAL3","AGE1", "FEM1", "AGE3", "FEM3"]].dropna()
            df = df[(df['AGE1'] >= min_age) & (df['AGE1'] < max_age) & (df['AGE3'] >= min_age) & (df['AGE3'] < max_age)]
            dfs.append(df)
            
        self.df_data = pd.concat(dfs)
        self.result_path = "result"
        self.variables = ["AGE1", "FEM1", "AGE3", "FEM3"]
        
    def draw_model_age(self, clf, 
                       age1:List, 
                       FEM1: int, 
                       age3: List, 
                       FEM3: int, 
                       flag: int, 
                       name: str,
                       path: str):
        """_summary_

        Args:
            clf (_type_): _description_
            age1 (List): _description_
            FEM1 (int): _description_
            age3 (List): _description_
            FEM3 (int): _description_
            flag (int): _description_
            name (str): _description_
            path (str): _description_
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if flag == 0:
            value = np.ones(age1.shape)
            fem1 = value * FEM1
            fem3 = value * FEM3
            for m in age3[::2]:
                data = np.array([age1, fem1, value * m, fem3]).transpose()
                prob = clf.predict_proba(data)
                ax.plot(age1,  prob[:, 1], label=m)
        else:
            value = np.ones(age3.shape)
            fem1 = value * FEM1
            fem3 = value * FEM3
            for m in age1[::2]:
                data = np.array([value * m, fem1, age3, fem3]).transpose()
                prob = clf.predict_proba(data)
                ax.plot(age3,  prob[:, 1], label=m)

        colormap = plt.cm.nipy_spectral #nipy_spectral, Set1,Paired   
        colors = [colormap(i) for i in np.linspace(0, 1,len(ax.lines))]
        for i,j in enumerate(ax.lines):
            j.set_color(colors[i])

        ax.set_xticks(np.arange(0, 100, 10))
        ax.set_yticks(np.arange(0, 1, 0.1))
        ax.legend(loc=2, fontsize=5)
        
        if FEM1 == 0:
            driver = 'Driver(M)'
        else:
            driver = 'Driver(F)'
            
        if FEM3 == 0:
            passager = 'Passenger(M)'
        else:
            passager = 'Passenger(F)'
            
        fname = os.path.join(path, name + f'-({driver}, {passager})')
            
        ax.set_title(fname)
        if flag == 0:
            ax.set_xlabel("Driver Age")
            fname = fname + '-driver'
        else:
            ax.set_xlabel("Passenger Age")
            fname = fname + '-passenger'
            
        ax.set_ylabel("Fatal Probability")
        ax.grid()
        fig.savefig(f"{fname}.png")
        
    def get_clf_data(self, clf, score, name):
            fatal_coef, fatal_intercept = clf.coef_[0], clf.intercept_[0]
            fatal = f"{name} = {fatal_intercept:.6f}"
            for i, v in enumerate(fatal_coef):
                if v > 0:
                    fatal = fatal + f" + {v:.6f} * {self.variables[i]}" 
                else:
                    fatal = fatal + f" - {-v:.6f} * {self.variables[i]}" 
            
            res = {
                "coeff": list(fatal_coef),
                "intercept": float(fatal_intercept),
                "train_score": score,
                "fomular": fatal
            }
            
            return res
        
    def build_model(self):
        """_summary_
        """
        X = self.df_data[self.variables].to_numpy()
        y1 = np.array(self.df_data["FATAL1"].to_list())
        y3 = np.array(self.df_data["FATAL3"].to_list())
        clf_fatal1, score1 = model_fit(X, y1)
        clf_fatal3, score3 = model_fit(X, y3)
        
        path = os.path.join(self.result_path, "overall")
        if not os.path.exists(path):
            os.makedirs(path)
            
        # fname = os.path.join(path, "logistic_model.json")
        # result = {
        #     "FATAL1": self.get_clf_data(clf_fatal1, score1, "FATAL1"),
        #     "FATAL3": self.get_clf_data(clf_fatal3, score3, "FATAL3")
        # }
        # with open(fname, "w") as f:
        #     json.dump(result, f, indent=4)
        
        age1 = np.arange(21, 97)
        age3 = np.arange(21, 97)
        sex = [0, 1]
        for fem1 in sex:
            for fem3 in sex:
                self.draw_model_age(clf_fatal1, age1, fem1, age3, fem3, 0, "FATAL1", path)
                self.draw_model_age(clf_fatal1, age1, fem1, age3, fem3, 1, "FATAL1", path)
                self.draw_model_age(clf_fatal3, age1, fem1, age3, fem3, 0, "FATAL3", path)
                self.draw_model_age(clf_fatal3, age1, fem1, age3, fem3, 1, "FATAL3", path)
                
    def model_age_group(self, win_size: int=18):
        """_summary_

        Args:
            win_size (int, optional): _description_. Defaults to 18.
        """
        start = self.min_age
        sex = [0, 1]
        model_data = list()
        
        for end in tqdm(range(19, self.max_age), desc="building models..."):
            df = self.df_data[(self.df_data['AGE1'] >= start) & (self.df_data['AGE1'] < end) & (self.df_data['AGE3'] >= start) & (self.df_data['AGE3'] < end)]
            X = df[self.variables].to_numpy()
            y1 = np.array(df["FATAL1"].to_list())
            y3 = np.array(df["FATAL3"].to_list())
            clf_fatal1, score1 = model_fit(X, y1)
            clf_fatal3, score3 = model_fit(X, y3)
            
            res = {
                "age_group": [start, end]
            }
            res["mixture"] = {
                "ave_age":[df['AGE1'].mean(), df['AGE3'].mean()],
                "train_score_fatal1": score1,
                "train_score_fatal3": score3,
                "FATAL1": self.get_clf_data(clf_fatal1, score1, "FATAL1"),
                "FATAL3": self.get_clf_data(clf_fatal3, score3, "FATAL3")
            }
            
            model_data.append(res)
            
            for fem1 in sex:
                for fem3 in sex:
                    df1 = df[(df['FEM1'] == fem1) & (df['FEM3'] == fem3)]
                    X = df1[self.variables].to_numpy()
                    y1 = np.array(df1["FATAL1"].to_list())
                    y3 = np.array(df1["FATAL3"].to_list())
                    clf_fatal1, score1 = model_fit(X, y1)
                    clf_fatal3, score3 = model_fit(X, y3)
                    res[f"{fem1}-{fem3}"] = {
                        "ave_age":[df1['AGE1'].mean(), df1['AGE3'].mean()],
                        "train_score_fatal1": score1,
                        "train_score_fatal3": score3,
                        "FATAL1": self.get_clf_data(clf_fatal1, score1, "FATAL1"),
                        "FATAL3": self.get_clf_data(clf_fatal3, score3, "FATAL3")
                    }
                
            model_data.append(res)
            
            if end - start >= win_size:
                start += 1
            end += 1
        
        
        path = os.path.join(self.result_path, "age_group")
        if not os.path.exists(path):
            os.makedirs(path)
        fname = os.path.join(path, f"logistic_model_age_group_{win_size}.json")
        with open(fname, "w") as f:
            json.dump(model_data, f, indent=4)
    
    
    def df_age_group_data(self, path: str, win_size: int=18):
        """_summary_

        Args:
            path (str): _description_
            win_size (int, optional): _description_. Defaults to 18.
        """
        cur_fname = f"logistic_model_age_group_{win_size}"
        fname = os.path.join(path, f"{cur_fname}.json")
        with open(fname) as f:
            age_group_data = json.load(f)
        
        sex = [0, 1]
        sheet_fields = ['mixture']
        mapping = {sheet_fields[0]: "Overall"}
        for fem1 in sex:
            for fem3 in sex:
                v = f"{fem1}-{fem3}"
                sheet_fields.append(v)
                name = "Male" if fem1 == 0 else "Female" + '-'
                name = name + ("Male" if fem3 == 0 else "Female")
                mapping[v] = name
        
        cols = ["Age group", 
                "Driver Age", "D_AGE1", "D_FEM1", "D_AGE3", "D_FEM3",
                "Passenger Age", "P_AGE1", "P_FEM1", "P_AGE3", "P_FEM3"]
        data = {name: list() for name in cols}
        
        df_dict = {name:deepcopy(data) for name in sheet_fields}
        
        for data in age_group_data:
            for name in sheet_fields:
                df_dict[name][cols[0]].append(str(data["age_group"][0]) + '-' + str(data["age_group"][1]))
                df_dict[name][cols[1]].append(data[name]["ave_age"][0])
                for i in range(4):
                    df_dict[name][cols[i+2]].append(data[name]["FATAL1"]["coeff"][i])
                df_dict[name][cols[6]].append(data[name]["ave_age"][1])
                for i in range(4):
                    df_dict[name][cols[i+7]].append(data[name]["FATAL3"]["coeff"][i])
                    
        dfs = list()
        sheet_names = list()
        for name, data in df_dict.items():
            dfs.append(pd.DataFrame.from_dict(data))
            sheet_names.append(mapping[name])
            
        save_multiple_dfs(dfs, sheet_names, os.path.join(path, f"{cur_fname}.xlsx"))
        
    def age_gender(self):
        """_summary_
        """
        df = self.df_data[(self.df_data['FATAL1']==2)|(self.df_data['FATAL3']==2)]
        X = df[self.variables].to_numpy()
        y1 = np.array(df["FATAL1"].to_list())
        y3 = np.array(df["FATAL3"].to_list())
        clf_fatal1, score1 = model_fit(X, y1)
        clf_fatal3, score3 = model_fit(X, y3)
        print(score1, score3)
        
        E_FTATL1 = sum(clf_fatal1.predict(X))
        E_FTATL3 = sum(clf_fatal3.predict(X))
        
        #Male driver
        X_M = np.copy(X)
        X_M[:, 1] = 0
        X_M[:, 0] += 10
        M1_FATAL1 = sum(clf_fatal1.predict(X_M))
        M1_FATAL3 = sum(clf_fatal3.predict(X_M))
        
        #Female Driver
        X_F = np.copy(X)
        X_F[:, 1] = 1
        F1_FATAL1 = sum(clf_fatal1.predict(X_F))
        F1_FATAL3 = sum(clf_fatal3.predict(X_F))
        
        print(E_FTATL1, E_FTATL3, M1_FATAL1, M1_FATAL3, F1_FATAL1, F1_FATAL3)
        
        
        
        
            

def main():
    data_path = "accident_fatal_data"
    years = [2019, 2020, 2021]
    prefix = "accident_fatal_"
    fars_model = FARS_Model(data_path, years, prefix=prefix)
    fars_model.build_model()
    # fars_model.model_age_group(win_size=18)
    # fars_model.df_age_group_data("result/age_group", win_size=18)
    #fars_model.age_gender()
    
if __name__ == '__main__':
    main()
