from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def get_confusion_matrix_one_hot_tc(runname,model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,                                                                                                                                                                                 
    where truth is 0/1, and max along each row of model_results is model result                                                                                                                                                                                      
    '''
    mr=[]
    mr2=[]
    mr3=[]
    for x in model_results:
        mr.append(np.argmax(x))
        mr2.append(x)
        if np.argmax(x)==0:
            mr3.append(1-x[np.argmax(x)])
        elif np.argmax(x)==1:
            mr3.append(x[np.argmax(x)])
    model_results=np.asarray(mr)
    truth=np.asarray(truth)
    cm=confusion_matrix(y_target=truth,y_predicted=np.rint(np.squeeze(model_results)),binary=True)
    fig,ax=plot_confusion_matrix(conf_mat=cm,figsize=(5,5))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('/users/exet4487/Figures/'+runname+'confmat.png')
    fpr,tpr,thresholds=roc_curve(truth,np.asarray(mr3))
    plt.figure()
    lw = 2
    aucval=auc(fpr,tpr)
    print(aucval)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % aucval)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('/users/exet4487/Figures/'+runname+'_roc.png')
    np.save('/users/exet4487/confmatdata/'+runname+'_fp.npy',fpr)
    np.save('/users/exet4487/confmatdata/'+runname+'_tp.npy',tpr)
    return cm
