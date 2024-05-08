import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


def plot(epochs,val_acc,train_acc,val_loss,train_loss):
    epochs=range(epochs)
    fig,axes=plt.subplots(1,2,figsize=(15,15))
    axes[0].plot(epochs,val_acc)
    axes[0].plot(epochs,train_acc)
    axes[0].set_title("Epochs vs Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(['Val accuracy','Train accuracy'])

    axes[1].plot(epochs,val_loss)
    axes[1].plot(epochs,train_loss)
    axes[1].set_title("Epochs vs Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend(['Val Loss','Train Loss'])
    plt.show()

def confusion_mat(true_labels,pred_labels):
    cm=confusion_matrix(true_labels,pred_labels)
    sns.set(font_scale=1.4) 
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', annot_kws={"size": 16})
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion matrix")
    plt.show()

def scores(true_labels,pred_labels):
    precision=precision_score(true_labels,pred_labels)
    recall=recall_score(true_labels,pred_labels)
    f1=f1_score(true_labels,pred_labels)
    print(f"Precision:{precision:.2f}\nRecall:{recall:.2f}\nF1 score:{f1:.2f}")

def roc_plot(true_labels,pred_probs):
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def prec_recall_curve(true_labels,pred_probs):
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    plt.figure()
    plt.plot(recall, precision, color='darkorange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.show()
