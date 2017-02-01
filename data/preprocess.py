import numpy as np
import sys
import os

DATA = 'gug_annotations.tsv'

data = np.loadtxt(DATA, delimiter='\t', dtype=object, skiprows=1)
#train_data = [d for d in data if d[5] == 'train']
#train_data = data[data[:, 5] == 'train']

train_data = data[np.logical_or.reduce([data[:, 5] == x for x in ['train', 'dev']])]
#dev_data = data[data[:, 5] == 'dev']
test_data = data[data[:, 5] == 'test']
#dev_data = [d for d in data if d[5] == 'dev']
#test_data = [d for d in data if d[5] == 'test']

def split_info(data, out_dir):
    sents = np.array(data[:, 1], dtype=str)
    exp_scores = np.array(data[:, 2], dtype=float)
    cs_scores = np.array([[float(s) for s in d.strip('[]').split(', ')] for d in data[:, 3]])
    cs_avg_scores = np.array(data[:, 4], dtype=float)

    try:
        os.makedirs(out_dir)
    except:
        pass
    
    np.savetxt(os.path.join(out_dir, 'sentences.tsv'), sents, fmt='%s')
    np.savetxt(os.path.join(out_dir, 'expert.tsv'), exp_scores)
    np.savetxt(os.path.join(out_dir, 'cs.tsv'), cs_scores)
    np.savetxt(os.path.join(out_dir, 'cs_avg.tsv'), cs_avg_scores)


split_info(train_data, 'train')
#split_info(dev_data, 'dev')
split_info(test_data, 'test')
