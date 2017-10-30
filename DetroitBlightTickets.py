import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

def load_data():
    print('loading data')
    
    #read the training and test data in as dataframes
    train_df = pd.read_csv('data/train.csv', encoding='latin-1',usecols=['ticket_id','fine_amount', 'discount_amount', 'judgment_amount', 'compliance'])
    test_df = pd.read_csv('data/test.csv', encoding='latin-1',usecols=['ticket_id','fine_amount', 'discount_amount', 'judgment_amount'])
    
    #set ticket_id as the index for both
    train_df.set_index('ticket_id', inplace=True)
    test_df.set_index('ticket_id', inplace=True)
    
    #drop entries for which compliance is "not responsible", i.e. those whose value is NaN instead of binary
    train_df.dropna(subset=['compliance'], inplace=True)
    return train_df, test_df

def learn_data(train_df,test_df):
    print('learning data')

    features = list(test_df.columns)
    response = ['compliance']

    X = train_df[features]
    Y = np.array(train_df[response]).ravel()
    
    Xmean = X.mean()
    Xstd = X.std()
    X = (X - Xmean)*(1/Xstd)
    X_test = (test_df - Xmean)*(1/Xstd)
    
    clf = RandomForestClassifier(max_depth=15)
    scores = cross_val_score(clf, X, Y, cv=5, scoring='roc_auc')
    print(scores)
    
    clf.fit(X, Y)
    y_pred = clf.predict_proba(X_test)
    
    result_sr = pd.Series(data=y_pred[:,1], index=np.array(test_df.index.values.tolist()), dtype="float32")
    result_sr.index.name = 'ticket_id'
    
    plot_validation_curve(X,Y)
    
    return result_sr

def plot_validation_curve(X,Y):
    param_range = np.linspace(2, 16, 8)
    print('range of parameter max_depth:',param_range)
    train_scores, test_scores = validation_curve(RandomForestClassifier(), X, Y, param_name="max_depth",
                                                 param_range=param_range,cv=5, scoring="roc_auc", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with RandomForestClassifier")
    plt.xlabel("max_depth")
    plt.ylabel("roc_auc")
    plt.ylim(0.7, 0.8)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.2,color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.2,color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    
def blight_model():
    train_df, test_df = load_data()
    bm = learn_data(train_df, test_df)
    
    res = 'Data type Test: '
    res += ['Failed: type(bm) should Series\n','Passed\n'][type(bm)==pd.Series]
    res += 'Data shape Test: '
    res += ['Failed: len(bm) should be 61001\n','Passed\n'][len(bm)==61001]
    res += 'Data Values Test: '
    res += ['Failed: all values should be in [0.,1.]\n','Passed\n'][all((bm<=1.) & (bm>=0.))]
    res += 'Data Values type Test: '
    res += ['Failed: bm.dtype should be float\n','Passed\n'][str(bm.dtype).count('float')>0]
    res += 'Index type Test: '
    res += ['Failed: type(bm.index) should be Int64Index\n','Passed\n'][type(bm.index)==pd.Int64Index]
    res += 'Index values type Test: '
    res += ['Failed: type(bm.index[0]) should be int64\n','Passed\n'][str(type(bm.index[0])).count("int64")>0]
    res += 'Output index shape test:'
    res += ['Failed, bm.index.shape should be (61001,)\n','Passed\n'][bm.index.shape==(61001,)]
    res += 'Output index test: '
    if bm.index.shape==(61001,):
        res +=['Failed\n','Passed\n'][all(pd.read_csv('data/test.csv',usecols=[0],index_col=0).sort_index().index.values==bm.sort_index().index.values)]
    else:
        res+='Failed'
    print(res)

    return bm

blight_model()
