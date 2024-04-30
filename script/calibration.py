import numpy as np
import astrodash

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
    return aggregated_values

def extract_correct_softmax(filenames, redshift, trueclass, 
                            classifyHost=False, knownZ=True, 
                            smooth=6, rlapScores=True):
    classification = astrodash.Classify(filenames, redshift, classifyHost, 
                                        knownZ, smooth, rlapScores)

