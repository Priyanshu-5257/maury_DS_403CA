import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.metrics import mean_squared_error 

pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings('ignore')


df = pd.read_parquet('./../data/BMTC.parquet.gzip', engine='pyarrow') # This command loads BMTC data into a dataframe. 
                                                                      # In case of error, install pyarrow using: 
                                                                      # pip install pyarrow
dfInput = pd.read_csv('./../data/Input.csv',index_col="Unnamed: 0")
dfGroundTruth = pd.read_csv('./../data/GroundTruth.csv',index_col="Unnamed: 0") 
# NOTE: The file GroundTruth.csv is for participants to assess the performance their own codes

"""
CODE SUBMISSION TEMPLATE
1. The submissions should have the function EstimatedTravelTime().
2. Function arguments:
    a. df: It is a pandas dataframe that contains the data from BMTC.parquet.gzip
    b. dfInput: It is a pandas dataframe that contains the input from Input.csv
3. Returns:
    a. dfOutput: It is a pandas dataframe that contains the output
"""
def EstimatedTravelTime(df, dfInput): # The output of this function will be evaluated
    # Function body - Begins
    # Make changes here.
    test=dfInput.copy()
    dfOutput = pd.DataFrame()
    tstamp = df['Timestamp'].astype("string").str.split(' ',expand=True)
    tstamp=tstamp.drop(0,axis=1)
    tstamp=tstamp[1].astype("string").str.split(':',expand=True)
    tstamp=tstamp.astype(int)
    df["Time"]=tstamp[0]*60 + tstamp[1] + tstamp[2]*(1/60)
    df["Time"]=df["Time"]-df["Time"][0]
    df=df.drop("Timestamp",axis=1)
    busid=df["BusID"].unique()
    test_s=test.drop(["Dest_Lat","Dest_Long"],axis=1)
    test_d=test.drop(["Source_Lat","Source_Long"],axis=1)

    time_report=test.copy()
    time_report=time_report.drop(["Source_Lat","Source_Long","Dest_Lat","Dest_Long"],axis=1)

    dist_report=test.copy()
    dist_report=dist_report.drop(["Source_Lat","Source_Long","Dest_Lat","Dest_Long"],axis=1)

    dist=csd(test["Source_Lat"],test["Source_Long"],test["Dest_Lat"],test["Dest_Long"])
    tt=[]
    
    for i in busid:
        bus=df[(df["BusID"]==i)]
        X=bus.drop(["BusID","Speed","Time"],axis=1)
        bt=BallTree(X,metric='haversine')
        KNN=8
        if (len(bus)<KNN):
             KNN=len(bus)
        a,s=bt.query(test_s,k=KNN)
        a,d=bt.query(test_d,k=KNN)
    
        #time=np.array(bus["Time"])[d] - np.array(bus["Time"])[s]
        time=pd.DataFrame()
        DD=pd.DataFrame(np.array(bus["Time"])[d])
        SS=pd.DataFrame(np.array(bus["Time"])[s])
        for p in range(KNN):
            for q in range(KNN):
                time[(p+1)*(q+1)]=DD[p]-SS[q]
        time[(time<=0)]=100000
        time=time.T.min()
        time_report[i]=time
    
        lat_d=pd.DataFrame(np.array(bus["Latitude"])[d]).T.min()
        lat_s=pd.DataFrame(np.array(bus["Latitude"])[s]).T.min()
    
        long_d=pd.DataFrame(np.array(bus["Longitude"])[d]).T.min()
        long_s=pd.DataFrame(np.array(bus["Longitude"])[s]).T.min()    
    
        d_d=csd(lat_d,long_d,test["Dest_Lat"],test["Dest_Long"])
        d_s=csd(lat_s,long_s,test["Source_Lat"],test["Source_Long"])
        t_d=d_d + d_s
        dist_report[i]=t_d
    
    for k in range(len(test)):
        loc=loc=((5*dist_report.iloc[k] +1*dist_report.iloc[k]*time_report.iloc[k])==(5*dist_report.iloc[k] +1*dist_report.iloc[k]*time_report.iloc[k]).min())
        t=time_report.iloc[k][loc]    
        t=t+dist_report.iloc[k][loc]*6
        tt.append(np.array(t))
    dfOutput = np.array(tt)

    
    # Function body - Ends
    return dfOutput 
  
"""
Other function definitions here: BEGINS
"""
def csd(lat1, lon1, lat2, lon2, r=6371):

    coordinates = lat1, lon1, lat2, lon2
    phi1=np.radians(lat1)
    lambda1=np.radians(lon1)
    phi2=np.radians(lat2)
    lambda2=np.radians(lon2)
    a = (np.square(np.sin((phi2-phi1)/2)) + np.cos(phi1) * np.cos(phi2) * 
         np.square(np.sin((lambda2-lambda1)/2)))
    d = 2*r*np.arcsin(np.sqrt(a))
    return d
"""
Other function definitions here: ENDS
"""

dfOutput = EstimatedTravelTime(df, dfInput)