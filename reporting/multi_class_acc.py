from .bb_registry import register
from sklearn.metrics import accuracy_score


@register('multi_acc')
def model_accuracy(y, pred):
    return accuracy_score(y, pred)
