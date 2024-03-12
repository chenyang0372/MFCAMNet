import sklearn.metrics as metrics
def calculate_metrics(pred, label):
    # 计算auc和aupr
    roc_fpr, roc_tpr, roc_threshold = metrics.roc_curve(label, pred)
    roc_auc_1 = metrics.auc(roc_fpr, roc_tpr)
    roc_auc_2 = metrics.roc_auc_score(label, pred)
    auc = max(roc_auc_1, roc_auc_2)

    pr_precision, pr_recall, pr_threshold = metrics.precision_recall_curve(label, pred)
    ap_auc_1 = metrics.auc(pr_recall, pr_precision)
    ap_auc_2 = metrics.average_precision_score(label, pred)  # 另外一种计算方式 可能会和上面方法计算结果不一致，当数据点越来越多时，这种差距会减小
    aupr = max(ap_auc_1, ap_auc_2)

    # 计算accuracy、precision、recall和f1
    pred_type = [0 if score < 0.5 else 1 for score in pred]
    accuracy = metrics.accuracy_score(label, pred_type)
    precision = metrics.precision_score(label, pred_type)
    recall = metrics.recall_score(label, pred_type)
    f1 = metrics.f1_score(label, pred_type)

    aa.add(roc_fpr, roc_tpr, auc, pr_precision, pr_recall, aupr)    # 画图
    return {'auc': auc, 'aupr': aupr, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}