import os
import requests
import json
from zipfile import ZipFile, ZipInfo
from tqdm import tqdm
import pandas as pd


def download(path: str):
    """_summary_

    Args:
        path (str): _description_
    """
    years = list(range(1975, 2023))
    if not os.path.exists(path):
        os.makedirs(path)

    for year in tqdm(years, desc="Download ..."):
        url = f"https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip"
        resp = requests.get(url)
        if resp.status_code == 200:
            fname = os.path.join(path, f"FARS{year}NationalCSV.zip")
            with open(fname, 'bw') as f:
                f.write(resp.content)
        else:
            print(f"Connection Failed for Year {year}")


def unzip(path: str):
    """_summary_

    Args:
        path (str): _description_
    """
    files = [v for v in os.listdir(path) if v.endswith(".zip")]
    for file in tqdm(files, desc="Extracting..."):
        with ZipFile(os.path.join(path, file), 'r') as f:
            has_file = [os.path.dirname(v.filename) for v in f.infolist()]
            has_file = set(["file" for v in has_file if len(v) == 0])
            if len(has_file) > 0:
                f.extractall(os.path.join(path, os.path.splitext(file)[0]))
            else:
                f.extractall(path)
            
            
person_data_info = {
    "AGE":[
        {
            "Years": [1975, 2009],
            "age": [0, 97],
        },
        {
            "Years": [2009, 10000],
            "age": [0, 121],
        }
    ],
    "SEX":{
        "Male": 1,
        "Female": 2,
    },
    "PER_TYP":{
        "Driver": 1,
        "Passenger": 2
    },
    "INJ_SEV":{
        4: "Fatal Injury (K)",
        5: "Injured, Severity Unknown (U)",
        6: "Died Prior to Crash"
    },
    "SEAT_POS":[
        {
            "Years": [1975, 1982],
            1: "Front Left",
            2: "Front Middle",
            3: "Front Right",
            4: "Second Left",
            5: "Second Middle",
            6: "Second Right",
            10: "Front Seat - Other",
            20: "Second Seat - Other"
        },
        {
            "Years": [1982, 10000],
            11: "Front Left",
            12: "Front Middle",
            13: "Front Right",
            18: "Front Seat - Other",
            19: "Front Seat - Unknown",
            21: "Second Left",
            22: "Second Middle",
            23: "Second Right",
            28: "Second Seat - Other",
            29: "Second Seat - Unknown",
        }
    ]
}


def accident_fatal_data(path: str, year: int)->pd.DataFrame:
    """_summary_

    Args:
        path (str): _description_
        year (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    fname = os.path.join(path, f"FARS{year}NationalCSV", "person.csv")
    df_person = pd.read_csv(fname, dtype=str, encoding="ISO-8859-1")
    df_person = df_person.astype({"ST_CASE": int, "VEH_NO": int})
    
    fname = os.path.join(path, f"FARS{year}NationalCSV", "accident.csv")
    df_accident = pd.read_csv(fname, dtype=str, encoding="ISO-8859-1")
    vars = ["ST_CASE", "MONTH", "DAY", "HOUR", "MINUTE", "WEATHER", "LGT_COND"]
    df_accident = df_accident[vars].reset_index(drop=True).astype(int)
    
    accident_dict = df_accident.set_index('ST_CASE').to_dict(orient='index')

    fatal_age_data = {
        'Accident identifier (“ST_CASE”)': list(),
        'Vehicle identifier(“ST_CASE_V1”)': list(),
        "Model Year": list(),
        'FATAL1': list(),
        'AGE1': list(),
        'FEM1': list(),
        'FATAL3': list(),
        'AGE3': list(),
        'FEM3': list(),
    }
    
    for v in vars[1:]:
        fatal_age_data[v] = list()
    
    def is_front_seat(seat_pos, year):
        data =  person_data_info["SEAT_POS"][0]
        if year >= data["Years"][0] and year < data["Years"][1]:
            if seat_pos >= 1 and seat_pos <= 3 or seat_pos == 10:
                return True
        else:
            if seat_pos >= 11 and seat_pos < 20:
                return True
            
        return False

    groups = df_person.groupby(["ST_CASE", "VEH_NO"])
    for name, group in groups:
        st_case = int(name[0])
        veh_no = int(name[1])
        if veh_no == 0:
            continue
        
        model_year = None
        fatal, fata3, age1, age3, fem1, fem3 = None, None, None, None, None, None
        
        for i, row in group.iterrows():
            person_type = int(row["PER_TYP"])
            inj_type = int(row["INJ_SEV"])
            seat_pos = int(row['SEAT_POS'])
            age = int(row['AGE'])
            sex = int(row["SEX"])
            model_year = row['MOD_YEAR']
            if is_front_seat(seat_pos, year):
                if person_type == 1:
                    fatal = inj_type
                    age1 = age
                    if sex == 1:
                        fem1 = 0
                    else:
                        fem1 = 1
                else:
                    fata3 = inj_type
                    age3 = age
                    if sex == 1:
                        fem3 = 0
                    else:
                        fem3 = 1
        if fatal is None and fata3 is None:
            continue
        
        fatal_age_data["Accident identifier (“ST_CASE”)"].append(st_case)
        fatal_age_data["Vehicle identifier(“ST_CASE_V1”)"].append(veh_no)
        fatal_age_data["Model Year"].append(model_year)
        fatal_age_data["FATAL1"].append(fatal)
        fatal_age_data["FATAL3"].append(fata3)
        fatal_age_data["AGE1"].append(age1)
        fatal_age_data["AGE3"].append(age3)
        fatal_age_data["FEM1"].append(fem1)
        fatal_age_data["FEM3"].append(fem3)
        
        data = accident_dict[st_case]
        for v in vars[1:]:
            fatal_age_data[v].append(data[v])
        
    df = pd.DataFrame.from_dict(fatal_age_data)
    return df

def get_vehicle_data(vin: str, model_year: str):
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{vin}?format=json&modelyear={model_year}"
    resp = requests.get(url)
    if resp.status_code == 200:
        print(json.loads(resp.content))
    else:
        print(f"Connection Failed for VIN {vin}")


vin = "1HD1FCR16VY6"
model_year = "1999"
get_vehicle_data(vin, model_year)

#path = "FARS_data"
#download(path)
#unzip(path)
# years = list(range(2000,2023))

# result_path = "accident_data_full"
# if not os.path.exists(result_path):
#     os.makedirs(result_path)
    
# for year in tqdm(years, desc="processing..."):
#     fname = os.path.join(result_path, f"accident_{year}.xlsx")
#     if not os.path.exists(fname):
#         df = accident_fatal_data(path, year)
#         df.to_excel(fname, index=False)

#import note:
#get_vehicle_data get vehicle information, such as airbags
#PERSON table: MONTH, DAY, HOUR, MINUTE, MOD_YEAR, AGE, SEX, PER_TYPE, INJ_SEV, SEAT_POS, REST_USE, REST_MIS, AIR_BAG, 
#VEHICLE table: VIN