try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ModuleNotFoundError:
    import tensorflow as tf

from astrodash.multilayer_convnet import convnet_variables

## useful things to get a Frechet Inception distance inspired distance but using astrodash classifier for spectra
# design idea is basically use as much as possible from astrodash, 
# but hack the saved model and output instead of softmax sores, 
# output the second from last layer (right before the MLP classification head)


class RestoreModel(object):
    def __init__(self, modelFilename, inputImages, nw, nBins):
        self.reset()

        self.modelFilename = modelFilename
        self.inputImages = inputImages
        self.nw = nw
        self.nBins = nBins
        self.imWidthReduc = 8
        self.imWidth = 32  # Image size and width

        self.x, self.y_, self.keep_prob, self.y_conv, self.W, self.b = convnet_variables(self.imWidth,
                                                                                         self.imWidthReduc, self.nw,
                                                                                         self.nBins)

        self.saver = tf.train.Saver()

    def restore_variables(self):
        with tf.Session() as sess:
            self.saver.restore(sess, self.modelFilename)

            softmax = self.y_conv.eval(feed_dict={self.x: self.inputImages, self.keep_prob: 1.0})
            # print(softmax)

            # dwlog = np.log(10000. / 3500.) / self.nw
            # wlog = 3500. * np.exp(np.arange(0, self.nw) * dwlog)
            #
            # import matplotlib.pyplot as plt
            # spec = np.loadtxt('spec91Tmax_2006cz.txt')
            # weight = sess.run(self.W)[:, 23]
            # # plt.scatter(np.arange(1024), self.inputImages[0], marker='.', c=weight, cmap=plt.get_cmap('seismic'))
            # plt.scatter(wlog, spec, marker='.', c=weight, cmap=plt.get_cmap('seismic'))
            # plt.colorbar()

            # plt.imshow(sess.run(self.W)[:, 5].reshape([32,32]), cmap=plt.get_cmap('seismic'))
            # for i, classval in enumerate(np.arange(5, 306, 18)):
            #     plt.subplot(9, 2, i + 1)
            #     weight = sess.run(self.W)[:, classval]
            #     plt.title("{}, {}".format(i, classval))
            #     plt.imshow(weight.reshape([32, 32]), cmap=plt.get_cmap('seismic'))
            #     frame1 = plt.gca()
            #     frame1.axes.get_xaxis().set_visible(False)
            #     frame1.axes.get_yaxis().set_visible(False)
        return softmax

    def reset(self):
        tf.reset_default_graph()


class BestTypesListSingleRedshift(object):
    def __init__(self, modelFilename, inputImages, typeNamesList, nw, nBins):
        self.modelFilename = modelFilename
        self.inputImages = inputImages
        self.typeNamesList = typeNamesList
        self.nBins = nBins

        self.restoreModel = RestoreModel(self.modelFilename, self.inputImages, nw, nBins)
        self.typeNamesList = np.array(typeNamesList)

        # if more than one image, then variables will be a list of length len(inputImages)
        softmaxes = self.restoreModel.restore_variables()
        self.bestTypes, self.softmaxOrdered, self.idx = [], [], []
        for softmax in softmaxes:
            bestTypes, idx, softmaxOrdered = self.create_list(softmax)
            self.bestTypes.append(bestTypes)
            self.softmaxOrdered.append(softmaxOrdered)
            self.idx.append(idx)

    def create_list(self, softmax):
        idx = np.argsort(softmax)  # list of the index of the highest probabiites
        bestTypes = self.typeNamesList[idx[::-1]]  # reordered in terms of softmax probability columns

        return bestTypes, idx, softmax[idx[::-1]]
    


class Classify(object):
    def __init__(self, filenames=[], redshifts=[], smooth=6, minWave=3500, maxWave=10000, classifyHost=False,
                 knownZ=True, rlapScores=True, data_files='models_v06'):
        """ Takes a list of filenames and corresponding redshifts for supernovae.
        Files should contain a single spectrum, and redshifts should be a list of corresponding redshift floats
        """
        # download_all_files('v01')
        self.filenames = filenames
        self.redshifts = redshifts
        self.smooth = smooth
        self.minWave = minWave
        self.maxWave = maxWave
        self.classifyHost = classifyHost
        self.numSpectra = len(filenames)
        self.scriptDirectory = os.path.dirname(os.path.abspath(__file__))
        if knownZ and redshifts != []:
            self.knownZ = True
        else:
            self.knownZ = False
        if not self.redshifts:
            self.redshifts = [None] * len(filenames)
        self.rlapScores = rlapScores
        self.pars = get_training_parameters()
        self.nw, w0, w1 = self.pars['nw'], self.pars['w0'], self.pars['w1']
        self.dwlog = np.log(w1 / w0) / self.nw
        self.wave = w0 * np.exp(np.arange(0, self.nw) * self.dwlog)
        self.snTemplates, self.galTemplates = load_templates(
            os.path.join(self.scriptDirectory, data_files, 'models/sn_and_host_templates.npz'))

        if self.knownZ:
            if classifyHost:
                self.modelFilename = os.path.join(self.scriptDirectory, data_files,
                                                  "models/zeroZ_classifyHost/tensorflow_model.ckpt")
            else:
                self.modelFilename = os.path.join(self.scriptDirectory, data_files,
                                                  "models/zeroZ/tensorflow_model.ckpt")
        else:
            if self.classifyHost:
                raise ValueError("A model that classifies the host while simulatenously not knowing redshift does not "
                                 "exist currently. Please try one of the other 3 models or check back at a later "
                                 "date. Contact the author for support or further queries.")
                # self.modelFilename = os.path.join(self.scriptDirectory, data_files,
                #                                   "models/agnosticZ_classifyHost/tensorflow_model.ckpt")
            else:
                self.modelFilename = os.path.join(self.scriptDirectory, data_files,
                                                  "models/agnosticZ/tensorflow_model.ckpt")

    def _get_images(self, filename, redshift):
        if redshift in list(catalogDict.keys()):
            redshift = 0
        loadInputSpectra = LoadInputSpectra(filename, redshift, self.smooth, self.pars, self.minWave, self.maxWave,
                                            self.classifyHost)
        inputImage, inputRedshift, typeNamesList, nw, nBins, inputMinMaxIndex = loadInputSpectra.input_spectra()

        return inputImage, typeNamesList, nw, nBins, inputMinMaxIndex, inputRedshift

    def _input_spectra_info(self):
        inputImages = np.empty((0, int(self.nw)), np.float16)
        inputMinMaxIndexes = []
        for i in range(self.numSpectra):
            f = self.filenames[i]
            if self.knownZ:
                z = self.redshifts[i]
            else:
                z = 0
            inputImage, typeNamesList, nw, nBins, inputMinMaxIndex, inputRedshift = self._get_images(f, z)
            self.redshifts[i] = -inputRedshift[0]
            inputImages = np.append(inputImages, inputImage, axis=0)
            inputMinMaxIndexes.append(inputMinMaxIndex[0])
        bestTypesList = BestTypesListSingleRedshift(self.modelFilename, inputImages, typeNamesList, self.nw, nBins)
        bestTypes = bestTypesList.bestTypes
        softmaxes = bestTypesList.softmaxOrdered
        bestLabels = bestTypesList.idx

        return bestTypes, softmaxes, bestLabels, inputImages, inputMinMaxIndexes