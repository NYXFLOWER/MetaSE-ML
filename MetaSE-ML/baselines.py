import os
os.chdir('/home/flower/github/MetaSE-ML/MetaSE-ML')

from src.models import *
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#%%
# set fix seed for random numbers
np.random.seed(200)

# load data_gem and features
# GEM_name = "ML+GEM_Heart_version3"
GEM_name = 'ML+GEM_Lung_version1'
data_gem = MetaSEDataset(GEM_name)

# TODO: random exp
data_gem.fa = np.random.randn(data_gem.fa.shape[0]*data_gem.fa.shape[1]).reshape(data_gem.fa.shape)
data_gem.fi = np.random.randn(data_gem.fi.shape[0]*data_gem.fi.shape[1]).reshape(data_gem.fi.shape)


# #####################################################################################
# ########## train and test: from the side effect with the most drug samples ##########
# #####################################################################################
sample_count_by_se = np.array([len(i) for i in data_gem.data_by_labels.values()])
se_argsort = sample_count_by_se.argsort()

out = {'se_idx': [], 'se_name':[], 'n_sample': [], 'auroc_train': [], 'auroc_test': []}

se_list = list(data_gem.cid2seid.values())
se_cid_list = list(data_gem.cid2seid.keys())

# -------------------------------------------------------------------
# -- train and test a model for each side effect and record results
for i in range(se_argsort.shape[0]):
    # -- get the most common unseen side effect and samples
    label = se_argsort[-i]
    samples = data_gem.data_by_labels[label]

    if len(samples) < 10:
        continue

    print("se -- ", i)

    # -- train test splitting
    rng = np.random.default_rng()
    rng.shuffle(samples)

    n_samples = len(samples)
    split_rate = 0.9
    n_train = int(n_samples * split_rate)
    train_samples_pos, test_samples_pos = samples[:n_train], samples[n_train:]
    train_samples_neg, test_samples_neg = negative_sampling_by_label(train_samples_pos, data_gem.n_drug), negative_sampling_by_label(test_samples_pos, data_gem.n_drug)

    # -- prepare data_gem for training and testing
    train_samples = np.concatenate([train_samples_pos, train_samples_neg])
    test_samples = np.concatenate([test_samples_pos, test_samples_neg])

    train_label = np.concatenate([[1]*len(train_samples_pos), [0]*len(train_samples_neg)])
    test_label = np.concatenate([[1]*len(test_samples_pos), [0]*len(test_samples_neg)])

    train_feature = np.concatenate([data_gem.fi[train_samples], data_gem.fa[train_samples]], axis=1)
    test_feature = np.concatenate([data_gem.fi[test_samples], data_gem.fa[test_samples]], axis=1)

    # -- preprocessing and standardization
    scaler = StandardScaler()
    scaler.fit(train_feature)

    train_feature = scaler.transform(train_feature)
    test_feature = scaler.transform(test_feature)

    # model setting
    model = LinearSVC(random_state=0, tol=1e-05)

    # train
    model.fit(train_feature, train_label)

    train_pred = model.predict(train_feature)
    test_pred = model.predict(test_feature)


    fpr, tpr, thresholds = metrics.roc_curve(train_pred, train_label)

    auroc_train = metrics.roc_auc_score(train_label, train_pred)
    auroc_test = metrics.roc_auc_score(test_label, test_pred)
    # ap = metrics.average_precision_score(train_label, train_pred)   # average precision 

    # y, xx, _ = metrics.precision_recall_curve(train_label, train_pred)
    # auprc = metrics.auc(xx, y)

    # metrics.f1_score(train_label, train_pred, average='macro')
    # metrics.f1_score(train_label, train_pred, average='micro')
    # metrics.f1_score(train_label, train_pred, average='weighted')

    out['se_idx'].append(se_list[label])
    out['se_name'].append(data_gem.secid2name.get(se_cid_list[label]))
    out['n_sample'].append(n_samples)
    out['auroc_train'].append(auroc_train)
    out['auroc_test'].append(auroc_test)


out_df = pd.DataFrame(out)
out_df.to_csv(f'out/random_{GEM_name}_svmLinear_out.csv', index=False)
# %%
