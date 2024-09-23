from ucimlrepo import fetch_ucirepo 
import pandas as pd


# fetch dataset 
combined_cycle_power_plant = fetch_ucirepo(id=294) 
  
# data (as pandas dataframes) 
x = combined_cycle_power_plant.data.features 
y = combined_cycle_power_plant.data.targets 

#Dividing Data into Test, Validation and Training sets
test_x=x.iloc[8612:]
test_y=y.iloc[8612:]
train_x=[]
train_y=[]
val_x=[]
val_y=[]
for i in range(0,8612):
    if(i%5==0):
        val_x=x[i]
        val_y=y[i]
    else:
        train_x=x[i]
        train_y=y[i]
