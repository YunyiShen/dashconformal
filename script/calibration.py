import numpy as np
import astrodash
from tqdm import tqdm

#################################
### things related to calibration
#################################
def extract_just_class(stringarray):
    '''
    Args: a np.array of strings, that has form of '<something I want>: <something I don't want>'
    Return: a np.array with same size, just having '<something I want>'

    This is used to remove the age from astrodash, before we get the data. 
    '''
    def extract_desired_part(string): # this will just remove the age from astrodash, before I get the age data
        return string.split(":")[0].strip()
    vectorized_func = np.vectorize(extract_desired_part)
    return vectorized_func(stringarray)

def clean_labels(stringarray):
    return np.array([i.replace("IIL", "II").replace("-norm", "").replace("-csm","-CSM").replace("-broad", "-BL").replace("-02cx","x") for i in stringarray])



def aggregate_softmax(SNlabels, softmaxscore):
    '''
    this aggregate softmax scores based on class (since we remove ages)
    Args:
        SNlabels: np.array with just type labels, ages removed
        softmaxscore: softmax score with ages and type labels, will be aggregated based on type
    '''
    unique_labels = np.unique(SNlabels)
    aggregated_values = np.zeros_like(unique_labels, dtype=softmaxscore.dtype)
    for i, label in enumerate(unique_labels):
        indices = np.where(SNlabels == label)  # Get indices where label appears
        aggregated_values[i] = np.sum(softmaxscore[indices])  # Aggregate values
    return unique_labels, aggregated_values

def predict_just_class_for_one_star(filenames, redshift, 
                        classifyHost=False, knownZ=True, 
                        smooth=6, rlapScores=True, aggrclass = True):
    classification = astrodash.Classify([filenames], [redshift], classifyHost=classifyHost, 
                                        knownZ=knownZ, smooth = smooth, rlapScores = rlapScores)
    
    bestTypes, softmaxes, bestLabels, inputImages, inputMinMaxIndexes = classification._input_spectra_info()
    if aggrclass:
        SNlabels, aggr_softmax = aggregate_softmax(extract_just_class(bestTypes[0]), softmaxes[0])
        SNlabels = clean_labels(SNlabels)
        return SNlabels, aggr_softmax
    return bestTypes[0], softmaxes[0]

def extract_correct_softmax(filenames, redshift, trueclass, 
                            classifyHost=False, knownZ=True, 
                            smooth=6, rlapScores=True):
    '''
    This to extract softmax scores of the correct class in the validation set

    Args:
        filenames: either file name or a list of spectra
        redshift: list of known redshift (so assume for now)
        true class: list of true classes
        ... : other things for astrodash
    Value:
        a np.array with soft max scores of the right class
    '''
    assert len(filenames) == len(redshift)
    assert len(filenames) == len(trueclass)
    classification = astrodash.Classify(filenames, redshift, classifyHost = classifyHost, 
                                        knownZ = knownZ, smooth = smooth, rlapScores = rlapScores)
    
    bestTypes, softmaxes, bestLabels, inputImages, inputMinMaxIndexes = classification._input_spectra_info()

    out_softmax = []
    for i in range(len(trueclass)):
        this_star_types = clean_labels( extract_just_class(bestTypes[i]) )
        this_star_softmax = softmaxes[i]
        # accumulated soft max score
        out_softmax.append(np.sum(this_star_softmax[this_star_types == trueclass[i]]))

    return np.array(out_softmax)

def calculate_softmax_cutoff(softmax_scores, alpha = 0.1):
    '''
    Calculate softmax cutoff using observed correct softmax scores
    '''
    n = softmax_scores.shape[0]
    q_level = np.ceil((n+1)*(1.-alpha))/n
    #cal_scores = 1.- softmax_scores
    cal_scores = - np.log(softmax_scores + 1.e-22)
    qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    #print(np.log(softmax_scores))
    #print(np.log(cal_scores))
    #print(n, q_level, qhat)
    
    return - qhat

def conformal_cutoffs(valset,alpha = 0.1,classifyHost=False, knownZ=True, 
                      smooth=6, rlapScores=True):
    softmaxes = extract_correct_softmax(valset.spectra_list, valset.redshift_list,valset.type_list,
                                        classifyHost, knownZ, 
                                        smooth, rlapScores)
    cutoff = calculate_softmax_cutoff(softmaxes, alpha)
    return softmaxes, cutoff

#################################
### things related to set prediction
#################################

def top_class_set(scores, labels, alpha = 0.1, sorted = True):
    '''
    Take an array of softmax scores, find the indices that accumulated towards some threshold
    Args:
        scores: an array of softmax scores
        labels: an array of labels corresponding to softmax scores
        alpha: 1-threshold
        sorted: if the scores are sorted already in decending order
    Return:
        a list of labels that cummulate softmax scores to the threshold
    '''

    threshold = 1-alpha
    if not sorted:
        sorted_indices = np.argsort(scores)[::-1]  # Sort indices of scores array in descending order
    else:
        sorted = range(scores.shape[0])
    cumulative_sum = 0
    top_indices = []
    
    for idx in sorted_indices:
        cumulative_sum += scores[idx]
        top_indices.append(idx)
        if cumulative_sum >= threshold:
            break
    
    return labels[top_indices]


def conformal_set(scores, labels, thr):
    return labels[np.log(scores + 1.e-22) >= thr]


def make_set_predictions(testset, valset,alpha = 0.1,classifyHost=False, knownZ=True, 
                      smooth=6, rlapScores=True):
    _, conformal_cut = conformal_cutoffs(valset,alpha,classifyHost, knownZ, 
                      smooth, rlapScores)
    conformal_setpred = []
    topclass_setpred = []
    point_pred = []
    for i in tqdm(range(len(testset.spectra_list))):
        predlabels, aggr_softmax = predict_just_class_for_one_star(testset.spectra_list[i],
                                                                   testset.redshift_list[i], classifyHost, 
                                                                   knownZ, smooth, rlapScores)
        point_pred.append(predlabels[np.argmax(aggr_softmax)])
        conformal_setpred.append(conformal_set(aggr_softmax, predlabels, conformal_cut))
        topclass_setpred.append(top_class_set(aggr_softmax, predlabels, alpha, False))

    return testset.type_list, point_pred, conformal_setpred, topclass_setpred, conformal_cut
    



#################################
### things related to check
#################################
def is_covered(setpred, true_labels):
    assert len(true_labels) == len(setpred)
    return np.array([np.any(setpred[i] == true_labels[i]) for i in range(len(true_labels))])
