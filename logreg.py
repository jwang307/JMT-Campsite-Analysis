from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# import pandas as pd
# import seaborn as sns

'''
def print_confusion_matrix(confusion_matrix,
                           axes,
                           class_names,
                           fontsize=14):
    df_cm = pd.DataFrame(confusion_matrix,
                         index=class_names,
                         columns=class_names)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=0,
                                 ha='right',
                                 fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=45,
                                 ha='right',
                                 fontsize=fontsize)
    axes.set_ylabel('Truth')
    axes.set_xlabel('Predicted')
    axes.set_title('Confusion Matrix')
'''


def train_logreg(X, y):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X, y)
    return model


def test_logreg(model: LogisticRegression, X, y):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    # f1 = f1_score(y, y_pred)
    print(cm)
    print(accuracy)
    print(report)
    # print(f1)
    # print_confusion_matrix(cm, [])

