import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    'Customers':['Salman','Ali','Anas','Hamza','usman','anees','zain','arbab','faizan','shehrayar'],
    'Age':[19,25,30,23,34,50,45,40,23,28],
    'spending':[100,200,350,120,360,700,580,400,120,200]
}

df=pd.DataFrame(data)


X=df[['Age','spending']]

model=KMeans(n_clusters=2,random_state=42,n_init=10)

df['Group']=model.fit_predict(X)

plt.figure(figsize=(6,6))

for group in df['Group'].unique():
    group_Data=df[df['Group']==group]
    plt.scatter(group_Data['Age'],
                group_Data['spending'],
                label=f'Group{group}'
                
)

plt.xlabel('AGE')
plt.ylabel('SPENDING')
plt.title('Customer segmenation IN K-MEANS')
plt.grid(True)
plt.legend()
plt.show()
print(df)


