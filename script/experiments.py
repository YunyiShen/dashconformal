from dataloader import spectra_redshift_type
from calibration import is_covered, conformal_cutoffs,  make_set_predictions
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from json_tricks import dump


# Problem setup
alpha = 0.2 # 1-alpha is the desired coverage
split_por = 0.4

my_full_dataset = spectra_redshift_type()
my_full_dataset.readin_spectra(maxread=10000)
mycal, mytest = my_full_dataset.get_split(split_por)

val_softmax, conformalthr = conformal_cutoffs(mycal, alpha)
print((conformalthr))
plt.hist(np.log(np.array(val_softmax)+1e-22))
plt.savefig("softmax.jpg")

true_type, point, conformal, topclass, conformal_cut = make_set_predictions(mytest, mycal,alpha)

covering_conformal = is_covered(conformal, true_type)
covering_topclass = is_covered(topclass, true_type)

dumping = {"true_type": true_type, 
           "point": point,
           "conformal": conformal,
           "topclass": topclass, 
           "star": mytest.file_names_list, 
           "spectra": mytest.spectra_list, 
           "calibration_soft_max": val_softmax}
with open("experimental_results.json", "w") as file:
    dump(dumping, fp = file)
print(np.mean(np.array(covering_conformal)), np.mean(np.array(covering_topclass )))

