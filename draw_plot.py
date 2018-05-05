from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

result_dir = './test_result'

def main():
    models = sys.argv[1:]
    for model in models:
        y_true = np.load(os.path.join(result_dir, model +'_label' + '.npy')) 
        y_scores = np.load(os.path.join(model + '_output' + '.npy'))
        y_true = np.reshape(y_true,(-1))
        y_scores = np.reshape(y_scores,(-1))
    
        precision, recall, threshold = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, lw=2, label=model)
       
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'pr_curve'))

if __name__ == "__main__":
    main()
