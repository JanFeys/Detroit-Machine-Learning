import pandas as pd
import numpy as np

def load_data():
    print('loading data')
    
    #read the training and test data in as dataframes
    train_df = pd.read_csv('data/train.csv', encoding='latin-1',usecols=['ticket_id','fine_amount', 'discount_amount', 'judgment_amount', 'compliance'])
    test_df = pd.read_csv('data/test.csv', encoding='latin-1',usecols=['ticket_id','fine_amount', 'discount_amount', 'judgment_amount'])
    
    #set ticket_id as the index for both
    train_df.set_index('ticket_id', inplace=True)
    test_df.set_index('ticket_id', inplace=True)
    
    #drop entries for which compliance is "not responsible", i.e. those whose value is not 0 or 1 but NaN
    train_df.dropna(subset=['compliance'], inplace=True)
        
    ##some diagnostics:
     #output all columns
    print(list(train_df.columns.values))
    # #print datatypes for the columns
    #for col in train_df.columns:
    #    print(train_df[col].dtype, col)
     #count and print the occurrences of number in a column
    #vc = train_df.disposition.value_counts(dropna=False)
    #print(vc)
     #print out columns and shape of dataframe for info
    #print(list(train_df.columns.values))
     #print shape of the data
    #print(train_df.shape)
    #print(test_df.shape)
    
    return train_df, test_df

def process_data(train_df,test_df):
    print('processing data')
    
    return train_df, test_df

def learn_data(train_df,test_df):
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    print('learning data')

    features = list(test_df.columns)
    response = ['compliance']

    X = train_df[features]
    Y = np.array(train_df[response]).ravel()
    
    Xmean = X.mean()
    Xstd = X.std()
        
    X = (X - Xmean)*(1/Xstd)
    X_test = (test_df - Xmean)*(1/Xstd)
    
    #['fine_amount', 'discount_amount', 'judgment_amount', 'compliance']
    #vc = X_test.judgment_amount.value_counts(dropna=False)
    #print(vc)

    #train classifier
    clf = RandomForestClassifier(max_depth=25)
    
    scores = cross_val_score(clf, X, Y, cv=5, scoring='roc_auc')
    print(scores)
    print(np.mean(scores))
    
    #predict
    clf.fit(X, Y)
    y_pred = clf.predict_proba(X_test)
    
    result_sr = pd.Series(data=y_pred[:,1], index=np.array(test_df.index.values.tolist()), dtype="float32")
    result_sr.index.name = 'ticket_id'
    #print(result_sr)

    return result_sr
    
def blight_model():
    train_df, test_df = load_data()
    result_df = learn_data(train_df, test_df)
    print('done!')
    
    bm = result_df
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

    return result_df

blight_model()