"""
Models that employ an average of word embeddings as inputs.

"""
import numpy as np
import sys
import os
import argparse
import json

from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import sklearn.preprocessing as pp
from scipy.stats.stats import pearsonr

import GPy
import util
import flakes


############
# Parse args

parser = argparse.ArgumentParser()
parser.add_argument('inputs', help='the file with the sentence inputs')
parser.add_argument('labels', help='the file with the valence labels')
parser.add_argument('embs', help= 'the word embeddings file')
parser.add_argument('model', help='one of: "linear",' +
                    '"rbf", "mat32", "mat52", "ratquad", "mlp"')
parser.add_argument('label_preproc', help='one of "none", "scale", "warp"')
parser.add_argument('output_dir', help='directory where outputs will be stored')
parser.add_argument('--data_size', help='size of dataset, default is full size',
                    default=10000, type=int)
#parser.add_argument('--ard', help='set this flag to enable ARD', action='store_true')
args = parser.parse_args()

###################
# Load data

embs, index = util.load_embs_matrix(args.embs)
# X = []
# with open(args.inputs) as f:
#     for line in f:
#         sent = util.preprocess_sent(line.split('_')[1])
#         indices = util.get_indices(sent, index)
#         X.append([indices])
X = []
with open(args.inputs) as f:
    for line in f:
        X.append(util.preprocess_sent(line))
Y = np.loadtxt(args.labels)[:, None]

###################
# Preprocessing

#X = util.pad_sents(X)
#print X[:10]
#print Y[:10]
X = np.array([[x] for x in X], dtype=object)
data = np.concatenate((X, Y), axis=1)[:args.data_size]
np.random.seed(1000)
np.random.shuffle(data)
X = data[:, :-1]
Y = data[:, -1:]
#print data
#print X
#print Y

##############
# Get folds
kf = KFold(n_splits=10)
folds = kf.split(data)

##############
# Create output structure
main_out_dir = os.path.join(args.output_dir, 'sk', args.model, args.label_preproc)

#############
# Train models and report
fold = 0
for i_train, i_test in folds:
    X_train = X[i_train]
    #Y_train = np.array(Y[i_train], dtype=float)
    Y_train = Y[i_train]
    X_test = X[i_test]
    #Y_test = np.array(Y[i_test], dtype=float)
    Y_test = Y[i_test]

    # Scale Y if asked for
    if args.label_preproc == "scale":
        Y_scaler = pp.StandardScaler()
        Y_scaler.fit(Y_train)
        Y_train = Y_scaler.transform(Y_train)

    # Train model
    if args.model == 'linear':
        k = flakes.wrappers.gpy.GPyStringKernel(gap_decay=0.1, match_decay=0.1, order_coefs=[1.0] * 5, 
                                                embs=embs, device='/cpu:0', mode='tf-batch', 
                                                batch_size=1000, sim='dot', index=index,
                                                wrapper='none')
        #k = k * GPy.kern.Bias(1)
    elif args.model == 'norm':
        k = flakes.wrappers.gpy.GPyStringKernel(gap_decay=0.1, match_decay=0.1, order_coefs=[1.0] * 5, 
                                                embs=embs, device='/cpu:0', mode='tf-batch', 
                                                batch_size=1000, sim='dot', index=index,
                                                wrapper='norm')
        k = k * GPy.kern.Bias(1)
    elif args.model == 'arccos0':
        k = flakes.wrappers.gpy.GPyStringKernel(gap_decay=0.1, match_decay=0.1, order_coefs=[1.0] * 5, 
                                                embs=embs, device='/cpu:0', mode='tf-batch', 
                                                batch_size=1000, sim='dot', index=index,
                                                wrapper='arccos0')
        k = k * GPy.kern.Bias(1)
        #k['string.variance'].constrain_fixed(1e-1)
        #k['string.variance'].constrain_bounded(1e-4, 1)
        #k['string.gap'].constrain_bounded(0.01, 1)
        #k['string.match'].constrain_bounded(0.01, 1)
    elif args.model == 'rbf':
        k = flakes.wrappers.gpy.RBFStringKernel(gap_decay=0.1, match_decay=1e-1, order_coefs=[0.1] * 5,
                                                embs=embs, device='/cpu:0', mode='tf-batch', 
                                                batch_size=1000, sim='dot', 
                                                wrapper='none')
        #k['gap_decay'].constrain_bounded(0.0, 0.5)
        #k['match_decay'].constrain_bounded(0.0, 0.5)
        #k['coefs'].constrain_fixed(0.00001)

    k = k + GPy.kern.Bias(1)
    if args.label_preproc == "warp":
        model = GPy.models.WarpedGP(X_train, Y_train, kernel=k)
        model['warp_tanh.psi'] = np.random.lognormal(0, 1, (3, 3))
    else:
        #ll = GPy.likelihoods.Gaussian(variance=10.0)
        model = GPy.models.GPRegression(X_train, Y_train, kernel=k)#, noise_var=2.0)
        #model = GPy.core.GP(X_train, Y_train, kernel=k, likelihood=ll)
    #print model.checkgrad(verbose=True)
    #model.randomize()
    #model['Gaussian_noise.variance'] = 10
    model.optimize(messages=True, max_iters=100)
    print model
    print model['.*coefs.*']
    # Get predictions
    info_dict = {}
    #preds, vars = model.predict_noiseless(X_test)
    #print X_test
    preds, vars = model.predict(X_test)
    #print model.predict(X_test)
    if args.label_preproc == 'scale':
        preds = Y_scaler.inverse_transform(preds)
    info_dict['mae'] = MAE(preds, Y_test)
    info_dict['rmse'] = np.sqrt(MSE(preds, Y_test))
    info_dict['pearsonr'] = pearsonr(preds.flatten(), Y_test.flatten())
    lpd = model.log_predictive_density(X_test, Y_test)
    info_dict['nlpd'] = -np.mean(lpd)

    # Get parameters
    print model
    param_names = model.parameter_names()
    for p_name in param_names:
        if p_name == 'warp_tanh.psi':
            info_dict[p_name] = list([list(pars) for pars in model[p_name]])
        else:
            try:
                info_dict[p_name] = float(model[p_name])
            except TypeError: #ARD
                info_dict[p_name] = list(model[p_name])
    info_dict['log_likelihood'] = float(model.log_likelihood())
    # print model
    # info_dict['gap_decay'] = float(model['.*string.gap_decay'])
    # info_dict['match_decay'] = float(model['.*string.match_decay'])
    # info_dict['coefs'] = list(model['.*string.coefs'])
    # info_dict['noise'] = float(model['Gaussian_noise.variance'])
    # info_dict['log_likelihood'] = float(model.log_likelihood())
    # if args.model != 'linear':
    #     info_dict['variance'] = float(model['rbf_string.variance'])
    # if args.label_preproc == 'warp':
    #     info_dict['warp_psi'] = list([list(pars) for pars in model['warp_tanh.psi']])
    #     info_dict['warp_d'] = float(model['warp_tanh.d'])

    # Save information
    fold_dir = os.path.join(main_out_dir, str(fold))
    try:
        os.makedirs(fold_dir)
    except OSError:
        # Already exists
        pass
    with open(os.path.join(fold_dir, 'info.json'), 'w') as f:
        json.dump(info_dict, f, indent=2)
    #print preds
    #print vars
    np.savetxt(os.path.join(fold_dir, 'preds.tsv'), preds)
    np.savetxt(os.path.join(fold_dir, 'vars.tsv'), vars)

    # Cleanup
    #model.kern._implementation.sess.close()
    del model

    # Next fold
    fold += 1
    break
