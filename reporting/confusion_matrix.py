from .bb_registry import register
from sklearn.metrics import confusion_matrix


@register('confusion_matrix')
def report(y, pred):
    return confusion_matrix(y, pred)
