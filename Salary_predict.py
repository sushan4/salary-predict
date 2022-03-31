import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
global level
global company_type
global company_size
global exp_p
from tkinter import * 
df = pd.read_csv('it_salary.csv')
print(df.head(10))

print(df.shape)

print(df.info())
print(df.isnull().sum())
print(df.describe().T)
print(df.level.value_counts())
print(df.company_size.value_counts())
print(df.company_type.value_counts())
print(df[df['salary'] == df.salary.max()])

plt.figure(figsize=(12,6))
sns.boxplot(x=df.level, y=df.salary)

plt.figure(figsize=(12,6))
sns.boxplot(x=df.company_size, y=df.salary)

plt.figure(figsize=(12,6))
sns.boxplot(x=df.company_type, y=df.salary)

plt.figure(figsize=(10,8))
sns.scatterplot(x = 'yrs_exp', y = 'salary', data = df)

plt.figure(figsize=(10,8))
sns.scatterplot(x = 'yrs_exp', y = 'salary', hue = 'level', data = df)

data = df.copy()

ndata = pd.get_dummies(data, prefix_sep='_')
print(ndata.head())
print(ndata.shape)

print(ndata.info())

ndata = ndata.drop(['level_Junior','company_size_less than 50','company_type_Corporation'], axis='columns')
print(ndata.shape)

print(ndata.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

X = ndata.drop('salary',axis=1)

y = ndata.salary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=50)

lr = LinearRegression()
print(lr.fit(X_train, y_train))
print("Linear Regression R^2 Score: {:.4f}%".format(lr.score(X_test, y_test)*100))
y_pred = lr.predict(X_test)

print("Linear Regression RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

diff = y_test - y_pred
print(pd.DataFrame(np.c_[y_test , y_pred , diff] , columns=['Actual','Predicted','Difference']))

print(X_train.iloc[792])
print(y_train.iloc[792])

value = lr.predict([[2,0,1,0,1,0,0,1,0,0]])[0]
print(value)
print(ndata.head())

#=============================================== prediction
def get_predicted_salary():
    global level
    global company_type
    global company_size
    global exp_p
   
    #yrs_exp = exp_p
    level_Head, level_Middle, level_Senior = 0, 0, 0
    company_size_100_1000, company_size_50_100, company_size_more_than_1000 = 0, 0, 0
    company_type_Agency, company_type_Product, company_type_Startup = 0, 0, 0
    if level == "Head":
        level_Head = 1
    elif level == "Middle":
        level_Middle = 1
    elif level == "Senior":
        level_Senior = 1

    if company_size == "100-1000":
        company_size_100_1000 = 1
    elif company_size == "50-100":
        company_size_50_100 = 1
    elif company_size == "more_than_1000":
        company_size_more_than_1000 = 1

    if company_type == "Agency":
        company_type_Agency = 1
    elif company_type == "Product":
        company_type_Product = 1
    elif company_type == "Startup":
        company_type_Startup = 1

    x = [exp_p, level_Head, level_Middle, level_Senior, company_size_100_1000, company_size_50_100, company_size_more_than_1000, company_type_Agency, company_type_Product, company_type_Startup]
    value1 = str(int(lr.predict([x])[0]))[:-3] + "000"  # converting 123456.556767 to 123000
    messagebox.showinfo('PREDICTION','salary='+str(value1))

def select_exp():
    global exp_p
    exp_p=exp.get()
    print(exp_p)


def select_level():
    global level
    for i in l.curselection():
        level=l.get(i)
    print(level)


def select_size():
    global company_size
    for i in l1.curselection():
        company_size=l1.get(i)
    print(company_size)


def select_type():
    global company_type
    for i in l2.curselection():
        company_type=l2.get(i)
    print(company_type)
    
    
#==================================================
#======================GUI
import os
from tkinter.filedialog import askdirectory
import pygame

root=Tk()
root.title("AD1 SALARY PREDICTOR")
root.geometry("900x620+10+10")
root.config(bg='black')
#======================Tile and Subtitle 
y=Label(root,text='Salary Predictor',bg='black',
        font=('Open Sans',40,'bold'),fg='white')
y.pack()

#======================Tile and Subtitle 
z=Label(root,text='Kindly Select the option',bg='black',
        font=('Open Sans',12,'bold'),fg='white')
z.pack()

#======================Years of experience entry box
exp=Entry(root,width=15)
exp.place(x=600,y=140)
e=Label(root,text='Years of Experience: ',bg='black',font=('Open Sans',30,'bold'),fg='white')
e.place(x=180,y=120)
eb=Button(root,text="Done",bg="#e79700",width=10,height=1,
         font=('Open Sans',13,'bold'),fg='white',command=select_exp)
eb.place(x=730,y=130)
#======================level list box
l=Listbox(root,width=20,height=3,selectmode=SINGLE)

l.place(x=390,y=220)
l.insert(1,'Head')
l.insert(2,'Middle')
l.insert(3,'Senior')

a=Label(root,text='Level',bg='black',font=('Open Sans',30,'bold'),fg='white')
a.place(x=220,y=220)
s=Button(root,text="SELECT",bg="#e79700",width=20,height=1,
         font=('Open Sans',13,'bold'),fg='white',command=select_level)
s.place(x=580,y=220)


#=======================company size list box
l1=Listbox(root,width=20,height=3,selectmode=SINGLE)

l1.place(x=390,y=350)
l1.insert(1,'50-100')
l1.insert(2,'100-1000')
l1.insert(3,'More Than 1000')

#l1.config(yscrollcommand=scrollbar.set)
#scrollbar1.config(command=l.yview)
a1=Label(root,text='Company Size',bg='black',font=('Open Sans',30,'bold'),fg='white')
a1.place(x=35,y=350)
s1=Button(root,text="SELECT",bg="#e79700",width=20,height=1,
         font=('Open Sans',13,'bold'),fg='white',command=select_size)
s1.place(x=580,y=350)
company_size=''

#=======================company TYPE list box

l2=Listbox(root,width=20,height=3,selectmode=SINGLE)

l2.place(x=390,y=500)
l2.insert(1,'Agency')
l2.insert(2,'Product')
l2.insert(3,'Startup')



a2=Label(root,text='Company Type',bg='black',font=('Open Sans',30,'bold'),fg='white')
a2.place(x=35,y=500)
s2=Button(root,text="SELECT",bg="#e79700",width=20,height=1,
         font=('Open Sans',13,'bold'),fg='white',command=select_type)
s2.place(x=580,y=500)
#=========================

#=========================predict button
p=Button(root,text="PREDICT",bg="#e79700",width=20,height=1,
        font=('Open Sans',13,'bold'),fg='white',command=get_predicted_salary)
p.place(x=350,y=575)
root.mainloop()