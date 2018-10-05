import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import pickle
import time

from sklearn.decomposition import NMF
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

# read data
with open('../data/HCD35_pos.pickle', mode='rb') as fp:
    df_3 = pickle.load(fp)
    
with open('../data/HCD45_pos.pickle', mode='rb') as fp:
    df_4 = pickle.load(fp)

with open('../data/HCD65_pos.pickle', mode='rb') as fp:
    df_6 = pickle.load(fp)

print(df_3.shape)
print(df_4.shape)
print(df_6.shape)

# just data add to list
features = [
    df_3.drop('Subclass', axis=1),
    df_4.drop('Subclass', axis=1),
    df_6.drop('Subclass', axis=1)
]

t = pd.DataFrame()
n = [] 
for i in features:
    n_comp = 5
    
    # tSVD
    start = time.time()
    tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
    tsvd_results = tsvd.fit_transform(i)
    elapsed_time = time.time() - start
    print('tSVD: '+str(elapsed_time))
    n.append(['tSVD', elapsed_time])
    
    # PCA
    start = time.time()
    pca = PCA(n_components=n_comp, random_state=420)
    pca_results = pca.fit_transform(i)
    elapsed_time = time.time() - start
    print('PCA: '+str(elapsed_time))
    n.append(['PCA', elapsed_time])
    
    # ICA
    start = time.time()
    ica = FastICA(n_components=n_comp, random_state=420)
    ica_results = ica.fit_transform(i)
    elapsed_time = time.time() - start
    print('ICA: '+str(elapsed_time))
    n.append(['ICA', elapsed_time])
    
    # GRP
    start = time.time()
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
    grp_results = grp.fit_transform(i)
    elapsed_time = time.time() - start
    print('GRP: '+str(elapsed_time))
    n.append(['GRP', elapsed_time])
    
    # SRP
    start = time.time()
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
    srp_results = srp.fit_transform(i)
    elapsed_time = time.time() - start
    print('SRP: '+str(elapsed_time))
    n.append(['SRP', elapsed_time])
    
    # KPCA
    start = time.time()
    kpca = KernelPCA(n_components=n_comp, random_state=420)
    kpca_results = kpca.fit_transform(i)
    elapsed_time = time.time() - start
    print('KPCA: '+str(elapsed_time))
    n.append(['KPCA', elapsed_time])
    
    # TSNE
    start = time.time()
    tsne = TSNE(n_components=3, random_state=420) # ValueError: 'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree.
    tsne_results = tsne.fit_transform(i)
    elapsed_time = time.time() - start
    print('TSNE: '+str(elapsed_time))
    n.append(['TSNE', elapsed_time])
    
    # NMF
    start = time.time()
    nmf = NMF(n_components=n_comp, random_state=420)
    nmf_results = nmf.fit_transform(i)
    elapsed_time = time.time() - start
    print('NMF: '+str(elapsed_time))
    n.append(['NMF', elapsed_time])
    
    # RFAA (Recursive feature aggromeration algorithm)
    start = time.time()
    fag = FeatureAgglomeration(n_clusters=n_comp)
    fag_results = fag.fit_transform(i)
    elapsed_time = time.time() - start
    print('RFAA: '+str(elapsed_time))
    print(': '+str(elapsed_time))
    n.append(['RFAA', elapsed_time])
    
    # merge each data 
    t = pd.concat([
        t, 
        pd.DataFrame(tsvd_results),
        pd.DataFrame(pca_results),
        pd.DataFrame(ica_results),
        pd.DataFrame(grp_results),
        pd.DataFrame(srp_results),
        pd.DataFrame(kpca_results),
        pd.DataFrame(tsne_results),
        pd.DataFrame(nmf_results),
        pd.DataFrame(fag_results),
    ], axis=1)

f = [
    'tSVD', 'PCA', 'ICA','GRP',
    'SRP', 'KPCA'
]

# make column's name
v = []
for m in [3, 4, 6]:
    for i in f:
        for l in range(n_comp):
            v.append(i+'_'+str(l)+'_'+str(m))

p=['NMF', 'FAG']
q = []
for m in [3, 4, 6]:
    for i in p:
        for l in range(n_comp):
            q.append(i+'_'+str(l)+'_'+str(m))

p=['TSNE']
tsne = []
for m in [3, 4, 6]:
    for i in p:
        for l in range(3):
            tsne.append(i+'_'+str(l)+'_'+str(m))
    
    
t.columns = v+ tsne +q

# to csv
f = pd.concat([ df_3.Subclass, t], axis=1)
f.to_csv('../data/decomp_pos.csv')

# f=f.drop(['UMAP_0_3', 'UMAP_0_4', 'UMAP_0_6'], axis=1)

objective = f.Subclass
le = preprocessing.LabelEncoder()
objective = le.fit_transform(objective)

features = f.drop('Subclass', axis=1)

random_state=np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    objective,
    test_size=0.2
)

clf = rf()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

##### X Num 

g = {
    'n_comp':[1, 3, 5, 10, 20, 50, 100, 500,1000],
    'Accuracy':[
        0.7023809523809523, 0.7083333333333334, 0.7232142857142857, 0.7113095238095238, 
        0.7113095238095238, 0.6904761904761905, 0.6934523809523809, 0.6488095238095238, 0.6101190476190477
    ]
}
# n_comp 1 0.7023809523809523
# n_comp 3 0.7083333333333334
# n_comp 5 0.7232142857142857
# n_comp 10 0.7113095238095238
# n_comp 20 0.7113095238095238
# n_comp 50 0.6904761904761905
# n_comp 100 0.6934523809523809
# n_comp 500 0.6488095238095238
# n_comp 1000 0.6101190476190477

pd.DataFrame(g)

n

# Negative

with open('../data/HCD35_neg.pickle', mode='rb') as fp:
    df_3 = pickle.load(fp)
    
with open('../data/data/HCD45_neg.pickle', mode='rb') as fp:
    df_4 = pickle.load(fp)

with open('../data/HCD65_neg.pickle', mode='rb') as fp:
    df_6 = pickle.load(fp)

print(df_3.shape)
print(df_4.shape)
print(df_6.shape)

features = [
    df_3.drop('Subclass', axis=1),
    df_4.drop('Subclass', axis=1),
    df_6.drop('Subclass', axis=1)
]

t = pd.DataFrame()
n = [] 
for i in features:
    n_comp = 5
    
    # tSVD
    start = time.time()
    tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
    tsvd_results = tsvd.fit_transform(i)
    elapsed_time = time.time() - start
    print('tSVD: '+str(elapsed_time))
    n.append(['tSVD', elapsed_time])
    
    # PCA
    start = time.time()
    pca = PCA(n_components=n_comp, random_state=420)
    pca_results = pca.fit_transform(i)
    elapsed_time = time.time() - start
    print('PCA: '+str(elapsed_time))
    n.append(['PCA', elapsed_time])
    
    # ICA
    start = time.time()
    ica = FastICA(n_components=n_comp, random_state=420)
    ica_results = ica.fit_transform(i)
    elapsed_time = time.time() - start
    print('ICA: '+str(elapsed_time))
    n.append(['ICA', elapsed_time])
    
    # GRP
    start = time.time()
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
    grp_results = grp.fit_transform(i)
    elapsed_time = time.time() - start
    print('GRP: '+str(elapsed_time))
    n.append(['GRP', elapsed_time])
    
    # SRP
    start = time.time()
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
    srp_results = srp.fit_transform(i)
    elapsed_time = time.time() - start
    print('SRP: '+str(elapsed_time))
    n.append(['SRP', elapsed_time])
    
    # KPCA
    start = time.time()
    kpca = KernelPCA(n_components=n_comp, random_state=420)
    kpca_results = kpca.fit_transform(i)
    elapsed_time = time.time() - start
    print('KPCA: '+str(elapsed_time))
    n.append(['KPCA', elapsed_time])
    
    # TSNE
    start = time.time()
    tsne = TSNE(n_components=3, random_state=420) # ValueError: 'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree.
    tsne_results = tsne.fit_transform(i)
    elapsed_time = time.time() - start
    print('TSNE: '+str(elapsed_time))
    n.append(['TSNE', elapsed_time])
    
    # NMF
    start = time.time()
    nmf = NMF(n_components=n_comp, random_state=420)
    nmf_results = nmf.fit_transform(i)
    elapsed_time = time.time() - start
    print('NMF: '+str(elapsed_time))
    n.append(['NMF', elapsed_time])
    
    # FAG
    start = time.time()
    fag = FeatureAgglomeration(n_clusters=n_comp)
    fag_results = fag.fit_transform(i)
    elapsed_time = time.time() - start
    print('FAG: '+str(elapsed_time))
    n.append(['FAG', elapsed_time])
    
    # merge each data 
    t = pd.concat([
        t, 
        pd.DataFrame(tsvd_results),
        pd.DataFrame(pca_results),
        pd.DataFrame(ica_results),
        pd.DataFrame(grp_results),
        pd.DataFrame(srp_results),
        pd.DataFrame(kpca_results),
        pd.DataFrame(tsne_results),
        pd.DataFrame(nmf_results),
        pd.DataFrame(fag_results),
    ], axis=1)

f = [
    'tSVD', 'PCA', 'ICA','GRP',
    'SRP', 'KPCA'
]

# make column's name
v = []
for m in [3, 4, 6]:
    for i in f:
        for l in range(n_comp):
            v.append(i+'_'+str(l)+'_'+str(m))

p=['NMF', 'FAG']
q = []
for m in [3, 4, 6]:
    for i in p:
        for l in range(n_comp):
            q.append(i+'_'+str(l)+'_'+str(m))

p=['TSNE']
tsne = []
for m in [3, 4, 6]:
    for i in p:
        for l in range(3):
            tsne.append(i+'_'+str(l)+'_'+str(m))
    
    
t.columns = v+ tsne +q

# to csv
f = pd.concat([ df_3.Subclass, t], axis=1)
f.to_csv('../data/decomp_neg.csv')
