from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

def accuracy(output, target):
    return balanced_accuracy_score(target, output)


def precision(output, target):
    return precision_score(target, output, average='macro', zero_division=True)


def recall(output, target):
    return recall_score(target, output, average='macro', zero_division=True)


def f1(output, target):
    return f1_score(target, output, average='macro')