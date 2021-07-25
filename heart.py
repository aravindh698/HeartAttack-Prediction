import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
df=pd.read_csv(r'C:\Users\ARAVIND\Downloads\heart.csv')
scaler = StandardScaler()
scaler.fit(df)
from sklearn.model_selection import train_test_split
y=df['target']
X=df[['ca','oldpeak','cp','thalach']]
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=1)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb_predict=gnb.predict(X_test)
pickle.dump(gnb, open('model.pkl','wb'))
f = open('model.pkl','rb')
model1=pickle.load(f)
print(model1.predict([[0,1.4,145,0]]))
