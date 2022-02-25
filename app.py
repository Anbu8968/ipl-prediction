from google.protobuf.symbol_database import Default
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

st.title("IPL MATCHES PREDICTION 2020")

data=pd.read_csv("ipl.csv")
#print(data)
st.sidebar.subheader('Team Selection')

data=data[["city","team1","team2","toss_winner","toss_decision","winner"]]

team1=[]
for i in data["team1"]:
    if i not in team1:
        team1.append(i)
data.dropna(inplace=True)
for i in team1:
    x=team1.index(i)
    st.write(x,i)

data["team1"].replace("Mumbai Indians",0,inplace=True)
data["team1"].replace("Delhi Capitals",1,inplace=True)
data["team1"].replace("Kolkata Knight Riders",2,inplace=True)
data["team1"].replace("Rajasthan Royals",3,inplace=True)
data["team1"].replace("Kings XI Punjab",4,inplace=True)
data["team1"].replace("Royal Challengers Bangalore",5,inplace=True)
data["team1"].replace("Sunrisers Hyderabad",6,inplace=True)
data["team1"].replace("Chennai Super Kings",7,inplace=True)


data["team2"].replace("Mumbai Indians",0,inplace=True)
data["team2"].replace("Delhi Capitals",1,inplace=True)
data["team2"].replace("Kolkata Knight Riders",2,inplace=True)
data["team2"].replace("Rajasthan Royals",3,inplace=True)
data["team2"].replace("Kings XI Punjab",4,inplace=True)
data["team2"].replace("Royal Challengers Bangalore",5,inplace=True)
data["team2"].replace("Sunrisers Hyderabad",6,inplace=True)
data["team2"].replace("Chennai Super Kings",7,inplace=True)



# data["toss_winner"].replace("Royal Challengers Bangalore",0,inplace=True)
# data["toss_winner"].replace("Kings XI Punjab",1,inplace=True)
# data["toss_winner"].replace("Delhi Daredevils",2,inplace=True)
# data["toss_winner"].replace("Mumbai Indians",3,inplace=True)
# data["toss_winner"].replace("Kolkata Knight Riders",4,inplace=True)
# data["toss_winner"].replace("Rajasthan Royals",5,inplace=True)
# data["toss_winner"].replace("Deccan Chargers",6,inplace=True)
# data["toss_winner"].replace("Chennai Super Kings",7,inplace=True)
# data["toss_winner"].replace("Kochi Tuskers Kerala",8,inplace=True)
# data["toss_winner"].replace("Pune Warriors",9,inplace=True)
# data["toss_winner"].replace("Sunrisers Hyderabad",10,inplace=True)
# data["toss_winner"].replace("Gujarat Lions",11,inplace=True)
# data["toss_winner"].replace("Rising Pune Supergiants",12,inplace=True)
# data["toss_winner"].replace("Rising Pune Supergiant",13,inplace=True)
# data["toss_winner"].replace("Delhi Capitals",14,inplace=True)


data["winner"].replace("Mumbai Indians",0,inplace=True)
data["winner"].replace("Delhi Capitals",1,inplace=True)
data["winner"].replace("Kolkata Knight Riders",2,inplace=True)
data["winner"].replace("Rajasthan Royals",3,inplace=True)
data["winner"].replace("Kings XI Punjab",4,inplace=True)
data["winner"].replace("Royal Challengers Bangalore",5,inplace=True)
data["winner"].replace("Sunrisers Hyderabad",6,inplace=True)
data["winner"].replace("Chennai Super Kings",7,inplace=True)


x=data[["team1","team2"]].values
y=data["winner"].values



x_train,y_train=x,y

model=RandomForestClassifier(n_estimators=2,random_state=1)
model.fit(x_train,y_train)
#pre=model.predict(x_test)
# t1=int(input("entere t 1 : "))
# t2=int(input("rnter t2 : "))
#re=model.predict([[t1,t2]])


# for i in re:
#     


te1=int(st.sidebar.selectbox("Team 1",[0,1,2,3,4,5,6,7]))
te2=int(st.sidebar.selectbox("Team 2",[0,1,2,3,4,5,6,7],index=6))
print(type(te1))

re=model.predict([[te1,te2]])
if st.sidebar.button("Predict"):
    for i in re:
        # st.write(team1[te1],"VS",team1[te2]," finally The winner is : ",team1[i])
        st.sidebar.write("Winner : ",team1[i])
        #st.sidebar.header("The Model  is 55% Accuracy")

hide="""
<style>
#MaimMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}
</style>
"""
st.markdown(hide,unsafe_allow_html=True)
