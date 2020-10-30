import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE

con=sqlite3.connect('/Users/bhuvanagopalakrishnabasapur/PycharmProjects/Practise/Assignments/Wildfire_Project/Wildfire_project/FPA_FOD_20170508.sqlite')

df = pd.read_sql_query("SELECT * FROM Fires", con)

cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#print(cursor.fetchall())
cursor.close()
con.close()

def to_csv():
    con=sqlite3.connect('/Users/bhuvanagopalakrishnabasapur/PycharmProjects/Practise/Assignments/Wildfire_Project/Wildfire_project/FPA_FOD_20170508.sqlite')

    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        if (table_name == 'Fires'):
            table = pd.read_sql_query("SELECT * From {}".format(table_name), con)
            table.to_csv(table_name + '.csv', index_label='index')
    cursor.close()
    con.close()

to_csv()

print(df.columns)
#Index(['OBJECTID', 'FOD_ID', 'FPA_ID', 'SOURCE_SYSTEM_TYPE', 'SOURCE_SYSTEM',
    #    'NWCG_REPORTING_AGENCY', 'NWCG_REPORTING_UNIT_ID',
    #    'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT',
    #    'SOURCE_REPORTING_UNIT_NAME', 'LOCAL_FIRE_REPORT_ID',
    #    'LOCAL_INCIDENT_ID', 'FIRE_CODE', 'FIRE_NAME',
    #    'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID', 'MTBS_FIRE_NAME',
    #    'COMPLEX_NAME', 'FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY',
    #    'DISCOVERY_TIME', 'STAT_CAUSE_CODE', 'STAT_CAUSE_DESCR', 'CONT_DATE',
    #    'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'LATITUDE',
    #    'LONGITUDE', 'OWNER_CODE', 'OWNER_DESCR', 'STATE', 'COUNTY',
    #    'FIPS_CODE', 'FIPS_NAME', 'Shape'],
    #   dtype='object')


# Check for Nan values
def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

def percent_value_counts(df, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    ## creating a df with th
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    ## concating percent and total dataframe
    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)

print(missing_percentage(df))
#                              Total  Percent
# COMPLEX_NAME                1875282    99.72
# MTBS_ID                     1869462    99.41
# MTBS_FIRE_NAME              1869462    99.41
# ICS_209_INCIDENT_NUMBER     1854748    98.63
# ICS_209_NAME                1854748    98.63
# FIRE_CODE                   1555636    82.73
# LOCAL_FIRE_REPORT_ID        1459286    77.60
# CONT_TIME                    972173    51.70
# FIRE_NAME                    957189    50.90
# CONT_DOY                     891531    47.41
# CONT_DATE                    891531    47.41
# DISCOVERY_TIME               882638    46.94
# LOCAL_INCIDENT_ID            820821    43.65
# FIPS_NAME                    678148    36.06
# COUNTY                       678148    36.06
# FIPS_CODE                    678148    36.06
# NWCG_REPORTING_UNIT_NAME          0     0.00
# NWCG_REPORTING_UNIT_ID            0     0.00
# NWCG_REPORTING_AGENCY             0     0.00
# SOURCE_REPORTING_UNIT             0     0.00
# SOURCE_REPORTING_UNIT_NAME        0     0.00
# SOURCE_SYSTEM                     0     0.00
# SOURCE_SYSTEM_TYPE                0     0.00
# FPA_ID                            0     0.00
# FOD_ID                            0     0.00
# Shape                             0     0.00
# FIRE_YEAR                         0     0.00
# DISCOVERY_DATE                    0     0.00
# DISCOVERY_DOY                     0     0.00
# STAT_CAUSE_CODE                   0     0.00
# STAT_CAUSE_DESCR                  0     0.00
# FIRE_SIZE                         0     0.00
# FIRE_SIZE_CLASS                   0     0.00
# LATITUDE                          0     0.00
# LONGITUDE                         0     0.00
# OWNER_CODE                        0     0.00
# OWNER_DESCR                       0     0.00
# STATE                             0     0.00
# OBJECTID                          0     0.00

# Since the missing values of the following are very high, these attributes can be dropped
#COMPLEX_NAME, MTBS_ID, MTBS_FIRE_NAME, ICS_209_INCIDENT_NUMBER, ICS_209_NAME, FIRE_CODE, LOCAL_FIRE_REPORT_ID
df = df.drop(['COMPLEX_NAME', 'MTBS_ID', 'MTBS_FIRE_NAME', 
            'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'FIRE_CODE', 'LOCAL_FIRE_REPORT_ID'], axis=1)

# Will be removing the following columns as similar attributes are present with no missing values
#CONT_TIME, FIRE_NAME, CONT_DOY, CONT_DATE, DISCOVERY_TIME
df = df.drop(['CONT_TIME', 'CONT_DOY', 'CONT_DATE', 'DISCOVERY_TIME'], axis=1)

#Removing columns which do not affect the model 
#FIRE_NAME, LOCAL_INCIDENT_ID, FIPS_NAME , FIPS_CODE, NWCG_REPORTING_UNIT_NAME, NWCG_REPORTING_UNIT_ID,  
# NWCG_REPORTING_AGENCY, SOURCE_REPORTING_UNIT, SOURCE_REPORTING_UNIT_NAME, SOURCE_SYSTEM, SOURCE_SYSTEM_TYPE, 
# FPA_ID, FOD_ID, OWNER_CODE, OWNER_DESCR
# can remove COUNTY as well as there is an attribute STATE and cannot fill median values 
# for COUNTY as they may not match with the states

df = df.drop(['FIRE_NAME', 'LOCAL_INCIDENT_ID', 'FIPS_NAME' , 'FIPS_CODE', 'NWCG_REPORTING_UNIT_NAME', 'NWCG_REPORTING_UNIT_ID',  
                'NWCG_REPORTING_AGENCY', 'SOURCE_REPORTING_UNIT', 'SOURCE_REPORTING_UNIT_NAME', 'SOURCE_SYSTEM', 
                'SOURCE_SYSTEM_TYPE', 'FPA_ID', 'FOD_ID', 'OWNER_CODE', 'OWNER_DESCR', 'COUNTY'], axis=1)

print(missing_percentage(df))
#                   Total  Percent
# Shape                 0      0.0
# STATE                 0      0.0
# LONGITUDE             0      0.0
# LATITUDE              0      0.0
# FIRE_SIZE_CLASS       0      0.0
# FIRE_SIZE             0      0.0
# STAT_CAUSE_DESCR      0      0.0
# STAT_CAUSE_CODE       0      0.0
# DISCOVERY_DOY         0      0.0
# DISCOVERY_DATE        0      0.0
# FIRE_YEAR             0      0.0
# OBJECTID              0      0.0

print(df.columns)

#Index(['OBJECTID', 'FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY',
    #    'STAT_CAUSE_CODE', 'STAT_CAUSE_DESCR', 'FIRE_SIZE', 'FIRE_SIZE_CLASS',
    #    'LATITUDE', 'LONGITUDE', 'STATE', 'Shape'],
    #   dtype='object')
     
print(df.head(n=5))
   OBJECTID  FIRE_YEAR  DISCOVERY_DATE  DISCOVERY_DOY  ...   LATITUDE   LONGITUDE  STATE                                              Shape
0         1       2005       2453403.5             33  ...  40.036944 -121.005833     CA  b'\x00\x01\xad\x10\x00\x00\xe8d\xc2\x92_@^\xc0...
1         2       2004       2453137.5            133  ...  38.933056 -120.404444     CA  b'\x00\x01\xad\x10\x00\x00T\xb6\xeej\xe2\x19^\...
2         3       2004       2453156.5            152  ...  38.984167 -120.735556     CA  b'\x00\x01\xad\x10\x00\x00\xd0\xa5\xa0W\x13/^\...
3         4       2004       2453184.5            180  ...  38.559167 -119.913333     CA  b'\x00\x01\xad\x10\x00\x00\x94\xac\xa3\rt\xfa]...
4         5       2004       2453184.5            180  ...  38.559167 -119.933056     CA  b'\x00\x01\xad\x10\x00\x00@\xe3\xaa.\xb7\xfb]\...

[5 rows x 12 columns]

#Observed that the vales in 'Shape' attribute are very long and not understandable. 
# As the shape does not affect the model will be removing Shape as well
df = df.drop(['Shape'], axis=1)
print(df.head(n=5))

#    OBJECTID  FIRE_YEAR  DISCOVERY_DATE  DISCOVERY_DOY  STAT_CAUSE_CODE STAT_CAUSE_DESCR  FIRE_SIZE FIRE_SIZE_CLASS   LATITUDE   LONGITUDE STATE
# 0         1       2005       2453403.5             33              9.0    Miscellaneous       0.10               A  40.036944 -121.005833    CA
# 1         2       2004       2453137.5            133              1.0        Lightning       0.25               A  38.933056 -120.404444    CA
# 2         3       2004       2453156.5            152              5.0   Debris Burning       0.10               A  38.984167 -120.735556    CA
# 3         4       2004       2453184.5            180              1.0        Lightning       0.10               A  38.559167 -119.913333    CA
# 4         5       2004       2453184.5            180              1.0        Lightning       0.10               A  38.559167 -119.933056    CA


# STAT_CAUSE_DESCR is the description of STAT_CAUSE_CODE and similarly
# FIRE_SIZE_CLASS is the classification groups for FIRE_SIZE 
# can remove the columns STAT_CAUSE_DESCR and FIRE_SIZE_CLASS
df = df.drop(['STAT_CAUSE_DESCR', 'FIRE_SIZE_CLASS'], axis=1)
print(df.columns)

# Index(['OBJECTID', 'FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY',
#        'STAT_CAUSE_CODE', 'FIRE_SIZE', 'LATITUDE', 'LONGITUDE', 'STATE'],
#       dtype='object')

print(df.head())

#   OBJECTID  FIRE_YEAR  DISCOVERY_DATE  DISCOVERY_DOY  STAT_CAUSE_CODE  FIRE_SIZE   LATITUDE   LONGITUDE STATE
# 0         1       2005       2453403.5             33              9.0       0.10  40.036944 -121.005833    CA
# 1         2       2004       2453137.5            133              1.0       0.25  38.933056 -120.404444    CA
# 2         3       2004       2453156.5            152              5.0       0.10  38.984167 -120.735556    CA
# 3         4       2004       2453184.5            180              1.0       0.10  38.559167 -119.913333    CA
# 4         5       2004       2453184.5            180              1.0       0.10  38.559167 -119.933056    CA

#Changing pandas dataframe to numpy array
Y = df['FIRE_SIZE'].values
X = np.concatenate( (df['DISCOVERY_DATE'].values.reshape(-1,1),df['DISCOVERY_DOY'].values.reshape(-1,1), 
                     df['STAT_CAUSE_CODE'].values.reshape(-1,1),
                   df['LATITUDE'].values.reshape(-1,1), df['LONGITUDE'].values.reshape(-1,1)), axis = 1 )
print(Y[0:3])
print(X.shape)
#[0.1  0.25 0.1 ]
# (1880465, 5)

#Normalize the data
sc = StandardScaler()
X = sc.fit_transform(X)


# def MAPE(y_true,y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100