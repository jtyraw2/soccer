from numpy import average
import pandas as pd
import glob
import warnings

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

###TODO CALC DIFF, run models and report resutls
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

warnings.filterwarnings("ignore")
path = r"C:\Users\16306\Documents\soccer\data\*.csv"
main_df=[]
df3= pd.DataFrame()
df2= pd.DataFrame()
H= pd.DataFrame()
A= pd.DataFrame()
for fname in glob.glob(path):
   df=pd.read_csv(fname)
   df=df.iloc[:,0:23]
   df = df.drop(columns=['Referee', 'Div','Date','HTR','HF','AF','HC','AC','HY','AY'])
   main_df.append(df)
home = input ("Enter Home Team : ")
away = input("Enter Away Team : ")
for i in range(len(main_df)):
    reg_db= main_df[i][(main_df[i]['HomeTeam'] == home) & (main_df[i]['AwayTeam']== away)]
    rev_db=main_df[i][(main_df[i]['HomeTeam'] == away) & (main_df[i]['AwayTeam']== home)]
    df2 = df2.append(reg_db)
    df2 = df2.append(rev_db)
    A=A.append(reg_db)
    H=H.append(rev_db)

target = df2[['FTR']].replace(['A','H','D'],[0,1,2])
df2 = df2.drop(columns=['FTR'])

H=encode_and_bind(H, 'FTR')
A=encode_and_bind(A, 'FTR')


df3['Goal_Delta'] = (df2['HTHG'] - df2['HTAG'])
df3['shots_Delta'] = (df2['HS'] - df2['AS'])
df3['ST_delta'] = (df2['HST'] - df2['AST'])
df3['red_delta'] = (df2['HR'] - df2['AR'])

features = df3[['Goal_Delta', 'shots_Delta',
       'ST_delta', 'red_delta']]
X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.5, stratify = target)
names = ["Nearest Neighbors", "Logistic Regression","Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(5),
    LogisticRegression(),
    SVC(kernel="linear", C=0.025, probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000)]
cur_acc=0

for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        if accuracy > cur_acc:
            cur_acc=accuracy
            fit=clf.fit(X_train, y_train)
        print(name, accuracy)
df4= pd.DataFrame()
### some data set api request to grab data to potentially give you a live probability
res=[]
data_calculated = [{'Goal_Delta': 1, 'shots_Delta': -10, 'ST_delta':-12,'red_delta':-2}]  
df4 = pd.DataFrame(data_calculated)  
while(i<1000):
    X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.5, stratify = target)
    fit=clf.fit(X_train, y_train)
    prediction=fit.predict(df4)
    res.append(prediction)
    i=i+1
one_cnt=0
zero_cnt=0
two_cnt=0
print("Scenarios done fitting calculating win percentages")
for i in range(len(res)):
    if(res[i] == 0 ):
        zero_cnt=zero_cnt+1
    elif (res[i] == 1 ):
        one_cnt=one_cnt+1
    else:
        two_cnt=two_cnt+1
print("through 10,000 different random models the win/loss probabailities are :")
zero_prob=zero_cnt/len(res)
one_prob=one_cnt/len(res)
two_prob=two_cnt/len(res)
print(home," have a ",one_prob," % probability of winning")
print("the game has a ",two_prob," % probability of drawing")
print(away," have a ",zero_prob," % probability of winning")