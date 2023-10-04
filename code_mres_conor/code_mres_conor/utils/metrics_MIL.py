from __future__ import print_function, division
import numpy as np
from sklearn import metrics
# from Delong_CI import get_Delong_CI
import matplotlib.pyplot as plt


def calc_err_four_class(pred, real, pred_score, if_test):
    pred = np.array(pred)
    real = np.array(real)

    cm = metrics.confusion_matrix(real, pred)
    print('confusion matrix: ', cm)
    total = cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3]
    num_c = cm[0, 0]
    num_neu = cm[1, 1]
    num_non = cm[2, 2]
    num_4 = cm[3, 3]
    # num_5 = cm[4, 4]

    #tp_c = cm[0, 0]
    fp_c = cm[1, 0] + cm[2, 0] + cm[3, 0]
    tn_c = cm[1, 1] + cm[2, 2] +  cm[3, 3]  + cm[1, 2] + cm[2, 1] \
           + cm[1, 3] + cm[3, 1] + cm[2, 3] + cm[3, 2]

    #fn_c = cm[0, 1] + cm[0, 2] + cm[0, 3] + cm[0, 4]

    # tp_neu = cm[1, 1]
    fp_neu = cm[0, 1] + cm[2, 1] + cm[3, 1]
    # fn_neu = cm[1, 0] + cm[1, 2]
    tn_neu = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2] + cm[3, 3] \
           + cm[0, 3] + cm[3, 0] + cm[2, 3] + cm[3, 2]

    # tp_non = cm[2, 2]
    fp_non = cm[0, 2] + cm[1, 2] + cm[3, 2]
    # fn_non = cm[2, 0] + cm[2, 1]
    tn_non = cm[1, 1] + cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[3, 3] \
           + cm[0, 3] + cm[3, 0] + cm[1, 3] + cm[3, 1]

    # tp_non = cm[2, 2]
    fp_4 = cm[0, 3] + cm[1, 3] + cm[2, 3]
    # fn_non = cm[2, 0] + cm[2, 1]
    tn_4 = cm[1, 1] + cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[2, 2]  \
             + cm[0, 2] + cm[2, 0] + cm[1, 2] + cm[2, 1]

    # tp_non = cm[2, 2]
    # fp_5 = cm[0, 4] + cm[1, 4] + cm[2, 4] + cm[3, 4]
    # fn_non = cm[2, 0] + cm[2, 1]
    # tn_5 = cm[1, 1] + cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[2, 2] + cm[3, 3] \
    #        + cm[0, 2] + cm[2, 0] + cm[0, 3] + cm[3, 0] + cm[1, 2] + cm[2, 1] + cm[1, 3] + cm[3, 1] + cm[2, 3] + cm[
    #            3, 2]

    specificity_c = tn_c / (tn_c + fp_c)
    specificity_neu = tn_neu / (tn_neu + fp_neu)
    specificity_non = tn_non / (tn_non + fp_non)
    specificity_4 = tn_4 / (tn_4 + fp_4)
    # specificity_5 = tn_5 / (tn_5 + fp_5)

    # weighted spe for three classes
    spe = specificity_c * (num_c / total) \
          + specificity_neu * (num_neu / total) \
          + specificity_non * (num_non / total) +\
          specificity_4 * (num_4 / total)
          # + specificity_5 * (num_5 / total)# tn / tn + fp

    # print(pred_score)

    auc = metrics.roc_auc_score(y_true=real, y_score=pred_score, average='weighted', multi_class='ovr')



    precision_score = metrics.precision_score(real, pred, average='weighted')
    sen = metrics.recall_score(real, pred, average='weighted')  # tp / tp + fn
    f1_score = metrics.f1_score(real, pred, average='weighted')  # F1 = 2 * (precision * recall) / (precision + recall)

    if if_test is True:
        Compute_CI_four_class(pred, real, pred_score, sen, spe, f1_score, precision_score, auc)

    return precision_score, spe, sen, f1_score, cm, auc


def Compute_CI_four_class(y_pred, y_true, y_pred_score, Sen, Spe, F1_score, Precision_score, Auc):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_score = np.array(y_pred_score)

    n_bootstraps = 2000
    rng_seed = 42  # control reproducibility

    bootstrapped_spe = []
    bootstrapped_sen = []
    bootstrapped_f1 = []
    bootstrapped_pre = []
    bootstrapped_auc = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_true))
        if len(np.unique(y_true[indices])) < 4:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        ##y_pred = np.round(y_pred)
        cnf_matrix = metrics.confusion_matrix(y_true[indices], y_pred[indices])
        # print(len(y_true[indices]))
        # print(len(y_pred_score[indices]))
        # auc = metrics.roc_auc_score(y_true=y_true[indices], y_score=y_pred_score[indices], average='weighted', multi_class='ovo')

        total = cnf_matrix[0, 0] + cnf_matrix[1, 1] + cnf_matrix[2, 2] + cnf_matrix[3, 3]
        num_c = cnf_matrix[0, 0]
        num_neu = cnf_matrix[1, 1]
        num_non = cnf_matrix[2, 2]
        num_4 = cnf_matrix[3, 3]
        # num_5 = cnf_matrix[4, 4]

        # tp_c = cnf_matrix[0, 0]
        fp_c = cnf_matrix[1, 0] + cnf_matrix[2, 0] + cnf_matrix[3, 0]
        tn_c = cnf_matrix[1, 1] + cnf_matrix[2, 2] + cnf_matrix[3, 3] + cnf_matrix[1, 2] + cnf_matrix[2, 1] \
               + cnf_matrix[1, 3] + cnf_matrix[3, 1] + cnf_matrix[2, 3] + cnf_matrix[3, 2]

        # fn_c = cnf_matrix[0, 1] + cnf_matrix[0, 2] + cnf_matrix[0, 3] + cnf_matrix[0, 4]

        # tp_neu = cnf_matrix[1, 1]
        fp_neu = cnf_matrix[0, 1] + cnf_matrix[2, 1] + cnf_matrix[3, 1]
        # fn_neu = cnf_matrix[1, 0] + cnf_matrix[1, 2]
        tn_neu = cnf_matrix[0, 0] + cnf_matrix[0, 2] + cnf_matrix[2, 0] + cnf_matrix[2, 2] + cnf_matrix[3, 3] \
                 + cnf_matrix[0, 3] + cnf_matrix[3, 0] + cnf_matrix[2, 3] + cnf_matrix[3, 2]

        # tp_non = cnf_matrix[2, 2]
        fp_non = cnf_matrix[0, 2] + cnf_matrix[1, 2] + cnf_matrix[3, 2]
        # fn_non = cnf_matrix[2, 0] + cnf_matrix[2, 1]
        tn_non = cnf_matrix[1, 1] + cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[3, 3] \
                 + cnf_matrix[0, 3] + cnf_matrix[3, 0] + cnf_matrix[1, 3] + cnf_matrix[3, 1]

        # tp_non = cnf_matrix[2, 2]
        fp_4 = cnf_matrix[0, 3] + cnf_matrix[1, 3] + cnf_matrix[2, 3]
        # fn_non = cnf_matrix[2, 0] + cnf_matrix[2, 1]
        tn_4 = cnf_matrix[1, 1] + cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[2, 2] \
               + cnf_matrix[0, 2] + cnf_matrix[2, 0] + cnf_matrix[1, 2] + cnf_matrix[2, 1]

        # tp_non = cnf_matrix[2, 2]
        # fp_5 = cnf_matrix[0, 4] + cnf_matrix[1, 4] + cnf_matrix[2, 4] + cnf_matrix[3, 4]
        # fn_non = cnf_matrix[2, 0] + cnf_matrix[2, 1]
        # tn_5 = cnf_matrix[1, 1] + cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[2, 2] + cnf_matrix[3, 3] \
        #        + cnf_matrix[0, 2] + cnf_matrix[2, 0] + cnf_matrix[0, 3] + cnf_matrix[3, 0] + cnf_matrix[1, 2] + cnf_matrix[2, 1] + cnf_matrix[1, 3] + cnf_matrix[3, 1] + cnf_matrix[2, 3] + cnf_matrix[
        #            3, 2]

        specificity_c = tn_c / (tn_c + fp_c)
        specificity_neu = tn_neu / (tn_neu + fp_neu)
        specificity_non = tn_non / (tn_non + fp_non)
        specificity_4 = tn_4 / (tn_4 + fp_4)
        # specificity_5 = tn_5 / (tn_5 + fp_5)

        # weighted spe for three classes
        spe = specificity_c * (num_c / total) \
              + specificity_neu * (num_neu / total) \
              + specificity_non * (num_non / total) + \
              specificity_4 * (num_4 / total)

        auc = metrics.roc_auc_score(y_true=y_true[indices], y_score=y_pred_score[indices], average='weighted', multi_class='ovr')

        precision_score = metrics.precision_score(y_true[indices], y_pred[indices], average='weighted')
        sen = metrics.recall_score(y_true[indices], y_pred[indices], average='weighted')  # tp / tp + fn
        f1_score = metrics.f1_score(y_true[indices], y_pred[indices],
                                    average='weighted')  # F1 = 2 * (precision * recall) / (precision + recall)


        bootstrapped_spe.append(spe)
        bootstrapped_f1.append(f1_score)
        bootstrapped_auc.append(auc)

        bootstrapped_pre.append(precision_score)
        bootstrapped_sen.append(sen)

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.

    ##############################Control#################################
    """
    spe ci
    """
    # print(bootstrapped_spe)
    sorted_scores = np.array(bootstrapped_spe)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the specificity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Spe)

    """
    sen ci
    """
    sorted_scores = np.array(bootstrapped_sen)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the sensitivity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Sen)

    """
    auc ci
    """
    sorted_scores = np.array(bootstrapped_auc)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the auc score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper), Auc)

    # 'Delong method'
    # Delong_ci = get_Delong_CI(y_pred, y_true)
    # print('delong AUC CI : ', Delong_ci, Auc)


    ##############################Neuropathy#################################
    """
    f1 ci
    """
    sorted_scores = np.array(bootstrapped_f1)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the f1 : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                          confidence_upper),
        F1_score)

    """
    pre ci
    """
    sorted_scores = np.array(bootstrapped_pre)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the precision : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                 confidence_upper),
        Precision_score)



def calc_err_five_class(pred, real, pred_score, if_test):
    pred = np.array(pred)
    real = np.array(real)

    cm = metrics.confusion_matrix(real, pred)
    print('confusion matrix: ', cm)
    total = cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3] + cm[4, 4]
    num_c = cm[0, 0]
    num_neu = cm[1, 1]
    num_non = cm[2, 2]
    num_4 = cm[3, 3]
    num_5 = cm[4, 4]

    #tp_c = cm[0, 0]
    fp_c = cm[1, 0] + cm[2, 0] + cm[3, 0] + cm[4, 0]
    tn_c = cm[1, 1] + cm[2, 2] +  cm[3, 3] + cm[4, 4] + cm[1, 2] + cm[2, 1] \
           + cm[1, 3] + cm[3, 1] + cm[1, 4] + cm[4, 1] + cm[2, 3] + cm[3, 2] + cm[2, 4] + cm[4, 2] + cm[3, 4] + cm[4, 3]

    #fn_c = cm[0, 1] + cm[0, 2] + cm[0, 3] + cm[0, 4]

    # tp_neu = cm[1, 1]
    fp_neu = cm[0, 1] + cm[2, 1] + cm[3, 1] + cm[4, 1]
    # fn_neu = cm[1, 0] + cm[1, 2]
    tn_neu = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2] + cm[3, 3] + cm[4, 4] \
           + cm[0, 3] + cm[3, 0] + cm[0, 4] + cm[4, 0] + cm[2, 3] + cm[3, 2] + cm[2, 4] + cm[4, 2] + cm[3, 4] + cm[4, 3]

    # tp_non = cm[2, 2]
    fp_non = cm[0, 2] + cm[1, 2] + cm[3, 2] + cm[4, 2]
    # fn_non = cm[2, 0] + cm[2, 1]
    tn_non = cm[1, 1] + cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[3, 3] + cm[4, 4] \
           + cm[0, 3] + cm[3, 0] + cm[0, 4] + cm[4, 0] + cm[1, 3] + cm[3, 1] + cm[1, 4] + cm[4, 1] + cm[3, 4] + cm[4, 3]

    # tp_non = cm[2, 2]
    fp_4 = cm[0, 3] + cm[1, 3] + cm[2, 3] + cm[4, 3]
    # fn_non = cm[2, 0] + cm[2, 1]
    tn_4 = cm[1, 1] + cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[2, 2] + cm[4, 4] \
             + cm[0, 2] + cm[2, 0] + cm[0, 4] + cm[4, 0] + cm[1, 2] + cm[2, 1] + cm[1, 4] + cm[4, 1] + cm[2, 4] + cm[
                 4, 2]

    # tp_non = cm[2, 2]
    fp_5 = cm[0, 4] + cm[1, 4] + cm[2, 4] + cm[3, 4]
    # fn_non = cm[2, 0] + cm[2, 1]
    tn_5 = cm[1, 1] + cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[2, 2] + cm[3, 3] \
           + cm[0, 2] + cm[2, 0] + cm[0, 3] + cm[3, 0] + cm[1, 2] + cm[2, 1] + cm[1, 3] + cm[3, 1] + cm[2, 3] + cm[
               3, 2]

    specificity_c = tn_c / (tn_c + fp_c)
    specificity_neu = tn_neu / (tn_neu + fp_neu)
    specificity_non = tn_non / (tn_non + fp_non)
    specificity_4 = tn_4 / (tn_4 + fp_4)
    specificity_5 = tn_5 / (tn_5 + fp_5)

    # weighted spe for three classes
    spe = specificity_c * (num_c / total) \
          + specificity_neu * (num_neu / total) \
          + specificity_non * (num_non / total) +\
          specificity_4 * (num_4 / total)  \
          + specificity_5 * (num_5 / total)# tn / tn + fp

    # print(pred_score)

    auc = metrics.roc_auc_score(y_true=real, y_score=pred_score, average='weighted', multi_class='ovr')


    precision_score = metrics.precision_score(real, pred, average='weighted')
    sen = metrics.recall_score(real, pred, average='weighted')  # tp / tp + fn
    f1_score = metrics.f1_score(real, pred, average='weighted')  # F1 = 2 * (precision * recall) / (precision + recall)

    if if_test is True:
        Compute_CI_five_class(pred, real, pred_score, sen, spe, f1_score, precision_score, auc)

    return precision_score, spe, sen, f1_score, cm, auc


def Compute_CI_five_class(y_pred, y_true, y_pred_score, Sen, Spe, F1_score, Precision_score, Auc):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_score = np.array(y_pred_score)

    n_bootstraps = 2000
    rng_seed = 42  # control reproducibility

    bootstrapped_spe = []
    bootstrapped_sen = []
    bootstrapped_f1 = []
    bootstrapped_pre = []
    bootstrapped_auc = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_true))
        if len(np.unique(y_true[indices])) < 5:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        ##y_pred = np.round(y_pred)
        # print(y_true[indices])
        cnf_matrix = metrics.confusion_matrix(y_true[indices], y_pred[indices])
        # print(len(y_true[indices]))
        # print(len(y_pred_score[indices]))
        # auc = metrics.roc_auc_score(y_true=y_true[indices], y_score=y_pred_score[indices], average='weighted', multi_class='ovo')

        total = cnf_matrix[0, 0] + cnf_matrix[1, 1] + cnf_matrix[2, 2] + cnf_matrix[3, 3] + cnf_matrix[4, 4]
        num_c = cnf_matrix[0, 0]
        num_neu = cnf_matrix[1, 1]
        num_non = cnf_matrix[2, 2]
        num_4 = cnf_matrix[3, 3]
        num_5 = cnf_matrix[4, 4]

        # tp_c = cnf_matrix[0, 0]
        fp_c = cnf_matrix[1, 0] + cnf_matrix[2, 0] + cnf_matrix[3, 0] + cnf_matrix[4, 0]
        tn_c = cnf_matrix[1, 1] + cnf_matrix[2, 2] + cnf_matrix[3, 3] + cnf_matrix[4, 4] + cnf_matrix[1, 2] + cnf_matrix[2, 1] \
               + cnf_matrix[1, 3] + cnf_matrix[3, 1] + cnf_matrix[1, 4] + cnf_matrix[4, 1] + cnf_matrix[2, 3] + cnf_matrix[3, 2] + cnf_matrix[2, 4] + cnf_matrix[4, 2] + cnf_matrix[3, 4] + cnf_matrix[
                   4, 3]

        # fn_c = cnf_matrix[0, 1] + cnf_matrix[0, 2] + cnf_matrix[0, 3] + cnf_matrix[0, 4]

        # tp_neu = cnf_matrix[1, 1]
        fp_neu = cnf_matrix[0, 1] + cnf_matrix[2, 1] + cnf_matrix[3, 1] + cnf_matrix[4, 1]
        # fn_neu = cnf_matrix[1, 0] + cnf_matrix[1, 2]
        tn_neu = cnf_matrix[0, 0] + cnf_matrix[0, 2] + cnf_matrix[2, 0] + cnf_matrix[2, 2] + cnf_matrix[3, 3] + cnf_matrix[4, 4] \
                 + cnf_matrix[0, 3] + cnf_matrix[3, 0] + cnf_matrix[0, 4] + cnf_matrix[4, 0] + cnf_matrix[2, 3] + cnf_matrix[3, 2] + cnf_matrix[2, 4] + cnf_matrix[4, 2] + cnf_matrix[3, 4] + \
                 cnf_matrix[4, 3]

        # tp_non = cnf_matrix[2, 2]
        fp_non = cnf_matrix[0, 2] + cnf_matrix[1, 2] + cnf_matrix[3, 2] + cnf_matrix[4, 2]
        # fn_non = cnf_matrix[2, 0] + cnf_matrix[2, 1]
        tn_non = cnf_matrix[1, 1] + cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[3, 3] + cnf_matrix[4, 4] \
                 + cnf_matrix[0, 3] + cnf_matrix[3, 0] + cnf_matrix[0, 4] + cnf_matrix[4, 0] + cnf_matrix[1, 3] + cnf_matrix[3, 1] + cnf_matrix[1, 4] + cnf_matrix[4, 1] + cnf_matrix[3, 4] + \
                 cnf_matrix[4, 3]

        # tp_non = cnf_matrix[2, 2]
        fp_4 = cnf_matrix[0, 3] + cnf_matrix[1, 3] + cnf_matrix[2, 3] + cnf_matrix[4, 3]
        # fn_non = cnf_matrix[2, 0] + cnf_matrix[2, 1]
        tn_4 = cnf_matrix[1, 1] + cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[2, 2] + cnf_matrix[4, 4] \
               + cnf_matrix[0, 2] + cnf_matrix[2, 0] + cnf_matrix[0, 4] + cnf_matrix[4, 0] + cnf_matrix[1, 2] + cnf_matrix[2, 1] + cnf_matrix[1, 4] + cnf_matrix[4, 1] + cnf_matrix[2, 4] + cnf_matrix[
                   4, 2]

        # tp_non = cnf_matrix[2, 2]
        fp_5 = cnf_matrix[0, 4] + cnf_matrix[1, 4] + cnf_matrix[2, 4] + cnf_matrix[3, 4]
        # fn_non = cnf_matrix[2, 0] + cnf_matrix[2, 1]
        tn_5 = cnf_matrix[1, 1] + cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[2, 2] + cnf_matrix[3, 3] \
               + cnf_matrix[0, 2] + cnf_matrix[2, 0] + cnf_matrix[0, 3] + cnf_matrix[3, 0] + cnf_matrix[1, 2] + cnf_matrix[2, 1] + cnf_matrix[1, 3] + cnf_matrix[3, 1] + cnf_matrix[2, 3] + cnf_matrix[
                   3, 2]

        specificity_c = tn_c / (tn_c + fp_c)
        specificity_neu = tn_neu / (tn_neu + fp_neu)
        specificity_non = tn_non / (tn_non + fp_non)
        specificity_4 = tn_4 / (tn_4 + fp_4)
        specificity_5 = tn_5 / (tn_5 + fp_5)

        # weighted spe for three classes
        spe = specificity_c * (num_c / total) \
              + specificity_neu * (num_neu / total) \
              + specificity_non * (num_non / total) + \
              specificity_4 * (num_4 / total) \
              + specificity_5 * (num_5 / total)  # tn / tn + fp

        auc = metrics.roc_auc_score(y_true=y_true[indices], y_score=y_pred_score[indices], average='weighted', multi_class='ovr')

        precision_score = metrics.precision_score(y_true[indices], y_pred[indices], average='weighted')
        sen = metrics.recall_score(y_true[indices], y_pred[indices], average='weighted')  # tp / tp + fn
        f1_score = metrics.f1_score(y_true[indices], y_pred[indices],
                                    average='weighted')  # F1 = 2 * (precision * recall) / (precision + recall)


        bootstrapped_spe.append(spe)
        bootstrapped_f1.append(f1_score)
        bootstrapped_auc.append(auc)

        bootstrapped_pre.append(precision_score)
        bootstrapped_sen.append(sen)

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.

    ##############################Control#################################
    """
    spe ci
    """
    sorted_scores = np.array(bootstrapped_spe)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the specificity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Spe)

    """
    sen ci
    """
    sorted_scores = np.array(bootstrapped_sen)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the sensitivity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Sen)

    """
    auc ci
    """
    sorted_scores = np.array(bootstrapped_auc)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the auc score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper), Auc)

    # 'Delong method'
    # Delong_ci = get_Delong_CI(y_pred, y_true)
    # print('delong AUC CI : ', Delong_ci, Auc)


    ##############################Neuropathy#################################
    """
    f1 ci
    """
    sorted_scores = np.array(bootstrapped_f1)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the f1 : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                          confidence_upper),
        F1_score)

    """
    pre ci
    """
    sorted_scores = np.array(bootstrapped_pre)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the precision : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                 confidence_upper),
        Precision_score)




def calc_err_three_class(pred, real, pred_score, if_test):
    pred = np.array(pred)
    real = np.array(real)

    cm = metrics.confusion_matrix(real, pred)
    print('confusion matrix: ', cm)
    total = cm[0, 0] + cm[1, 1] + cm[2, 2]
    num_c = cm[0, 0]
    num_neu = cm[1, 1]
    num_non = cm[2, 2]

    tp_c = cm[0, 0]
    fp_c = cm[1, 0] + cm[2, 0]
    tn_c = cm[1, 1] + cm[2, 2] + cm[1, 2] + cm[2, 1]
    fn_c = cm[0, 1] + cm[0, 2]

    tp_neu = cm[1, 1]
    fp_neu = cm[0, 1] + cm[2, 1]
    fn_neu = cm[1, 0] + cm[1, 2]
    tn_neu = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]

    tp_non = cm[2, 2]
    fp_non = cm[0, 2] + cm[1, 2]
    fn_non = cm[2, 0] + cm[2, 1]
    tn_non = cm[1, 1] + cm[0, 0] + cm[0, 1] + cm[1, 0]

    specificity_c = tn_c / (tn_c + fp_c)
    specificity_neu = tn_neu / (tn_neu + fp_neu)
    specificity_non = tn_non / (tn_non + fp_non)

    # weighted spe for three classes
    spe = specificity_c * (num_c / total) \
          + specificity_neu * (num_neu / total) \
          + specificity_non * (num_non / total)  # tn / tn + fp

    # print(pred_score)

    auc = metrics.roc_auc_score(y_true=real, y_score=pred_score, average='weighted', multi_class='ovo')
    precision_score = metrics.precision_score(real, pred, average='weighted')
    sen = metrics.recall_score(real, pred, average='weighted')  # tp / tp + fn
    f1_score = metrics.f1_score(real, pred, average='weighted')  # F1 = 2 * (precision * recall) / (precision + recall)


    if if_test is True:
        Compute_CI_three_class(pred, real, pred_score, sen, spe, f1_score, precision_score, auc)

    return precision_score, spe, sen, f1_score, cm, auc


def Compute_CI_three_class(y_pred, y_true, y_pred_score, Sen, Spe, F1_score, Precision_score, Auc):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_score = np.array(y_pred_score)

    n_bootstraps = 2000
    rng_seed = 42  # control reproducibility

    bootstrapped_spe = []
    bootstrapped_sen = []
    bootstrapped_f1 = []
    bootstrapped_pre = []
    bootstrapped_auc = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        ##y_pred = np.round(y_pred)
        # print(y_true[indices])
        cnf_matrix = metrics.confusion_matrix(y_true[indices], y_pred[indices])
        # print(len(y_true[indices]))
        # print(len(y_pred_score[indices]))
        auc = metrics.roc_auc_score(y_true=y_true[indices], y_score=y_pred_score[indices], average='weighted', multi_class='ovo')

        total = cnf_matrix[0, 0] + cnf_matrix[1, 1] + cnf_matrix[2, 2]
        num_c = cnf_matrix[0, 0]
        num_neu = cnf_matrix[1, 1]
        num_non = cnf_matrix[2, 2]

        tp_c = cnf_matrix[0, 0]
        fp_c = cnf_matrix[1, 0] + cnf_matrix[2, 0]
        tn_c = cnf_matrix[1, 1] + cnf_matrix[2, 2] + cnf_matrix[1, 2] + cnf_matrix[2, 1]
        fn_c = cnf_matrix[0, 1] + cnf_matrix[0, 2]

        tp_neu = cnf_matrix[1, 1]
        fp_neu = cnf_matrix[0, 1] + cnf_matrix[2, 1]
        fn_neu = cnf_matrix[1, 0] + cnf_matrix[1, 2]
        tn_neu = cnf_matrix[0, 0] + cnf_matrix[0, 2] + cnf_matrix[2, 0] + cnf_matrix[2, 2]

        tp_non = cnf_matrix[2, 2]
        fp_non = cnf_matrix[0, 2] + cnf_matrix[1, 2]
        fn_non = cnf_matrix[2, 0] + cnf_matrix[2, 1]
        tn_non = cnf_matrix[1, 1] + cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0]

        specificity_c = tn_c / (tn_c + fp_c)
        specificity_neu = tn_neu / (tn_neu + fp_neu)
        specificity_non = tn_non / (tn_non + fp_non)

        # weighted spe for three classes
        spe = specificity_c * (num_c / total) \
              + specificity_neu * (num_neu / total) \
              + specificity_non * (num_non / total)  # tn / tn + fp

        precision_score = metrics.precision_score(y_true[indices], y_pred[indices], average='weighted')
        sen = metrics.recall_score(y_true[indices], y_pred[indices], average='weighted')  # tp / tp + fn
        f1_score = metrics.f1_score(y_true[indices], y_pred[indices],
                                    average='weighted')  # F1 = 2 * (precision * recall) / (precision + recall)


        bootstrapped_spe.append(spe)
        bootstrapped_f1.append(f1_score)
        bootstrapped_auc.append(auc)

        bootstrapped_pre.append(precision_score)
        bootstrapped_sen.append(sen)

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.

    ##############################Control#################################
    """
    spe ci
    """
    sorted_scores = np.array(bootstrapped_spe)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the specificity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Spe)

    """
    sen ci
    """
    sorted_scores = np.array(bootstrapped_sen)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the sensitivity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Sen)

    """
    auc ci
    """
    sorted_scores = np.array(bootstrapped_auc)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the auc score: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                  confidence_upper),
          Auc)

    # 'Delong method'
    # Delong_ci = get_Delong_CI(y_pred, y_true)
    # print('delong AUC CI : ', Delong_ci)


    ##############################Neuropathy#################################
    """
    f1 ci
    """
    sorted_scores = np.array(bootstrapped_f1)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the f1 : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                          confidence_upper),
        F1_score)

    """
    pre ci
    """
    sorted_scores = np.array(bootstrapped_pre)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the precision : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                 confidence_upper),
        Precision_score)




def calc_err_binary_class_breakdown(pred, real, pred_score, if_test):
    pred = np.array(pred)
    real = np.array(real)

    cm = metrics.confusion_matrix(real, pred)
    # print('confusion matrix: ', cm)
    # total = cm[0, 0] + cm[1, 1]
    # num_c = cm[0, 0]
    # num_neu = cm[1, 1]


    # tp_c = cm[0, 0]
    # fp_c = cm[1, 0]
    # tn_c = cm[1, 1]
    # fn_c = cm[0, 1] + cm[0, 2]

    tp_neu = cm[1, 1]
    fp_neu = cm[0, 1]
    fn_neu = cm[1, 0]
    tn_neu = cm[0, 0]

    # tp_non = cm[2, 2]
    # fp_non = cm[0, 2] + cm[1, 2]
    # fn_non = cm[2, 0] + cm[2, 1]
    # tn_non = cm[1, 1] + cm[0, 0] + cm[0, 1] + cm[1, 0]

    # specificity_c = tn_c / (tn_c + fp_c)
    spe = tn_neu / (tn_neu + fp_neu)
    sen = tp_neu / (tp_neu + fn_neu)
    precision_score = tp_neu / (tp_neu + fp_neu)
    f1_score = 2 * (precision_score * sen) / (precision_score + sen)
    auc = metrics.roc_auc_score(y_true=real, y_score=pred_score[:, 1])


    # precision_score = metrics.precision_score(real, pred, average='binary')
    # sen = metrics.recall_score(real, pred, average='binary')  # tp / tp + fn
    # f1_score = metrics.f1_score(real, pred, average='samples')  # F1 = 2 * (precision * recall) / (precision + recall)
    if if_test is True:
        Compute_CI_binary_class_breakdown(pred, real, pred_score, sen, spe, f1_score, precision_score, auc)

    return precision_score, spe, sen, f1_score, cm, auc


    # return precision_score, spe, sen, f1_score, cm

def Compute_CI_binary_class_breakdown(y_pred, y_true, y_pred_score, Sen, Spe, F1_score, Precision_score, Auc):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_score = np.array(y_pred_score)

    n_bootstraps = 2000
    rng_seed = 42  # control reproducibility

    bootstrapped_spe = []
    bootstrapped_sen = []
    bootstrapped_f1 = []
    bootstrapped_pre = []
    bootstrapped_auc = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        ##y_pred = np.round(y_pred)
        # print(y_true[indices])
        cnf_matrix = metrics.confusion_matrix(y_true[indices], y_pred[indices])
        auc = metrics.roc_auc_score(y_true=y_true[indices], y_score=y_pred_score[indices, 1], average='weighted')
        tp_neu = cnf_matrix[1, 1]
        fp_neu = cnf_matrix[0, 1]
        fn_neu = cnf_matrix[1, 0]
        tn_neu = cnf_matrix[0, 0]


        spe = tn_neu / (tn_neu + fp_neu)
        sen = tp_neu / (tp_neu + fn_neu)
        precision_score = tp_neu / (tp_neu + fp_neu)
        f1_score = 2 * (precision_score * sen) / (precision_score + sen)


        # precision_score = metrics.precision_score(y_true, y_pred, average='binary')
        # sen = metrics.recall_score(y_true, y_pred, average='binary')  # tp / tp + fn
        # f1_score = metrics.f1_score(y_true, y_pred,
        #                             average='binary')  # F1 = 2 * (precision * recall) / (precision + recall)


        bootstrapped_spe.append(spe)
        bootstrapped_f1.append(f1_score)
        bootstrapped_auc.append(auc)

        bootstrapped_pre.append(precision_score)
        bootstrapped_sen.append(sen)

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.

    ##############################Control#################################
    """
    spe ci
    """
    sorted_scores = np.array(bootstrapped_spe)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the specificity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Spe)

    """
    sen ci
    """
    sorted_scores = np.array(bootstrapped_sen)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the sensitivity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Sen)

    """
    auc ci
    """
    sorted_scores = np.array(bootstrapped_auc)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the auc score: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                  confidence_upper),
          Auc)

    'Delong method'
    # Delong_ci = get_Delong_CI(y_pred, y_true)
    # print('delong AUC CI : ', Delong_ci)

    ##############################Neuropathy#################################
    """
    f1 ci
    """
    sorted_scores = np.array(bootstrapped_f1)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the f1 : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                          confidence_upper),
        F1_score)

    """
    pre ci
    """
    sorted_scores = np.array(bootstrapped_pre)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the precision : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                 confidence_upper),
        Precision_score)




def calc_err_binary_class(pred, real, pred_score, if_test):
    pred = np.array(pred)
    real = np.array(real)

    cm = metrics.confusion_matrix(real, pred)
    print('confusion matrix: ', cm)
    # total = cm[0, 0] + cm[1, 1]
    # num_c = cm[0, 0]
    # num_neu = cm[1, 1]


    # tp_c = cm[0, 0]
    # fp_c = cm[1, 0]
    # tn_c = cm[1, 1]
    # fn_c = cm[0, 1] + cm[0, 2]

    tp_neu = cm[1, 1]
    fp_neu = cm[0, 1]
    fn_neu = cm[1, 0]
    tn_neu = cm[0, 0]

    # tp_non = cm[2, 2]
    # fp_non = cm[0, 2] + cm[1, 2]
    # fn_non = cm[2, 0] + cm[2, 1]
    # tn_non = cm[1, 1] + cm[0, 0] + cm[0, 1] + cm[1, 0]

    # specificity_c = tn_c / (tn_c + fp_c)
    spe = tn_neu / (tn_neu + fp_neu)
    sen = tp_neu / (tp_neu + fn_neu)
    precision_score = tp_neu / (tp_neu + fp_neu)
    f1_score = 2 * (precision_score * sen) / (precision_score + sen)
    auc = metrics.roc_auc_score(y_true=real, y_score=pred_score[:, 1], average='weighted')


    # precision_score = metrics.precision_score(real, pred, average='binary')
    # sen = metrics.recall_score(real, pred, average='binary')  # tp / tp + fn
    # f1_score = metrics.f1_score(real, pred, average='samples')  # F1 = 2 * (precision * recall) / (precision + recall)
    if if_test is True:
        Compute_CI_binary_class(pred, real, pred_score, sen, spe, f1_score, precision_score, auc)

    return precision_score, spe, sen, f1_score, cm, auc


    # return precision_score, spe, sen, f1_score, cm

def Compute_CI_binary_class(y_pred, y_true, y_pred_score, Sen, Spe, F1_score, Precision_score, Auc):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_score = np.array(y_pred_score)

    n_bootstraps = 2000
    rng_seed = 42  # control reproducibility

    bootstrapped_spe = []
    bootstrapped_sen = []
    bootstrapped_f1 = []
    bootstrapped_pre = []
    bootstrapped_auc = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        ##y_pred = np.round(y_pred)
        # print(y_true[indices])
        cnf_matrix = metrics.confusion_matrix(y_true[indices], y_pred[indices])
        auc = metrics.roc_auc_score(y_true=y_true[indices], y_score=y_pred_score[indices, 1], average='weighted')
        tp_neu = cnf_matrix[1, 1]
        fp_neu = cnf_matrix[0, 1]
        fn_neu = cnf_matrix[1, 0]
        tn_neu = cnf_matrix[0, 0]


        spe = tn_neu / (tn_neu + fp_neu)
        sen = tp_neu / (tp_neu + fn_neu)
        precision_score = tp_neu / (tp_neu + fp_neu)
        f1_score = 2 * (precision_score * sen) / (precision_score + sen)


        # precision_score = metrics.precision_score(y_true, y_pred, average='binary')
        # sen = metrics.recall_score(y_true, y_pred, average='binary')  # tp / tp + fn
        # f1_score = metrics.f1_score(y_true, y_pred,
        #                             average='binary')  # F1 = 2 * (precision * recall) / (precision + recall)


        bootstrapped_spe.append(spe)
        bootstrapped_f1.append(f1_score)
        bootstrapped_auc.append(auc)

        bootstrapped_pre.append(precision_score)
        bootstrapped_sen.append(sen)

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.

    ##############################Control#################################
    """
    spe ci
    """
    sorted_scores = np.array(bootstrapped_spe)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the specificity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Spe)

    """
    sen ci
    """
    sorted_scores = np.array(bootstrapped_sen)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the sensitivity: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                    confidence_upper),
          Sen)

    """
    auc ci
    """
    sorted_scores = np.array(bootstrapped_auc)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95 % Confidence interval for the auc score: [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                  confidence_upper),
          Auc)

    'Delong method'
    # Delong_ci = get_Delong_CI(y_pred, y_true)
    # print('delong AUC CI : ', Delong_ci)

    ##############################Neuropathy#################################
    """
    f1 ci
    """
    sorted_scores = np.array(bootstrapped_f1)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the f1 : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                          confidence_upper),
        F1_score)

    """
    pre ci
    """
    sorted_scores = np.array(bootstrapped_pre)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "95 % Confidence interval for the precision : [{:0.3f} - {:0.3}]".format(confidence_lower,
                                                                                 confidence_upper),
        Precision_score)



