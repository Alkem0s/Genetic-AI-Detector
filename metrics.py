# metrics.py
import tensorflow as tf

@tf.function
def calculate_precision_recall_f1(y_true, y_pred):
    """
    Optimized precision, recall, F1 calculation with fewer operations.
    """
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    # Calculate components in one pass
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    
    # Calculate precision and recall with single condition checks
    precision = tf.cond(
        tf.equal(tp + fp, 0),
        lambda: tf.cond(tf.equal(tf.reduce_sum(y_true_f), 0), lambda: 1.0, lambda: 0.0),
        lambda: tp / (tp + fp)
    )
    
    recall = tf.cond(
        tf.equal(tp + fn, 0),
        lambda: 1.0,
        lambda: tp / (tp + fn)
    )
    
    f1 = tf.cond(
        tf.equal(precision + recall, 0),
        lambda: 0.0,
        lambda: 2 * (precision * recall) / (precision + recall)
    )
    
    return precision, recall, f1

@tf.function
def calculate_balanced_accuracy(y_true, y_pred):
    """
    Optimized balanced accuracy calculation.
    """
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    tn = tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f))
    
    sensitivity = tf.cond(tf.equal(tp + fn, 0), lambda: 1.0, lambda: tp / (tp + fn))
    specificity = tf.cond(tf.equal(tn + fp, 0), lambda: 1.0, lambda: tn / (tn + fp))
    
    return (sensitivity + specificity) / 2.0