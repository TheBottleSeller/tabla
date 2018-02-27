import numpy as np
import scipy as sci
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.neighbors import KNeighborsClassifier

######THE FOLLOWING ARE DEFAULT SETTINGS FOR run_analysis()##########
freq_features = ['mean_mfcc_0',
                'mean_mfcc_1',
                'mean_mfcc_2',
                'mean_mfcc_3',
                'mean_mfcc_4',
                'mean_mfcc_5',
                'mean_mfcc_7',
                'mean_mfcc_8',
                'mean_mfcc_9',
                'mean_mfcc_10',
                'mean_mfcc_11',
                'mean_centroid']

visual_sample = ['mean_mfcc_3',
                'mean_mfcc_4',
                'mean_mfcc_5',
                'mean_mfcc_7',
                'mean_mfcc_8']

#####################################################################

#This is an Analysis class that
#will structure further statistical analyses.
#enter the data as a dataframe.
#build models and put them in Analysis.models dictionary

class Analysis():
    def __init__(self, data, class_id, models ={}):
        self.data = data
        self.class_id = class_id
        self.models = models

    #this neatly applies a univariate analysis to each feature.
    def single_variate_test(self, func, name):
        df0 = self.data[self.data[self.class_id] == 0]
        df1 = self.data[self.data[self.class_id] == 1]
        result_df = pd.DataFrame()
        for col in list(self.data):
            try:
                result = func(df0[col],df1[col])
                result_df[name + col] = [result]
            except:
                result_df[name + col] = [None]
                print("Statistical Test did not work for {0}".format(str(list(df0[col]))))
        return(result_df)

    #this creates a pca in the Analysis.models attribute
    def add_pca(self, features = [], n_comp = 3):

        if features !=[]:
            df = self.data.loc[:,features]
        else:
            features = [x for x in list(self.data) if x != self.class_id]
            df = self.data.loc[:,features]

        pca = PCA(n_components = n_comp)
        pca.fit(df)
        print("Explained variance by principle components")
        print(pca.explained_variance_ratio_)
        self.models['pca'] = pca

    #this trains a knn for the analysis.
    #it only uses the subset of features you provide it.
    def train_knn(self, features, k_neighbors):

        if features !=[]:
            ftrain_x = self.data.loc[:,features]
        else:
            features = [x for x in list(self.data) if x != self.class_id]
            ftrain_x = self.data.loc[:,features]

        neigh = KNeighborsClassifier(n_neighbors = k_neighbors)
        neigh.fit(ftrain_x, np.ravel(self.data.loc[:,[self.class_id]]))
        self.models['knn'] = neigh

    #this function applies k_means clustering to the data.
    #most of it was taken from SoundAnalysis.py
    def k_means_cluster(self, features):
        ftrArr = np.array(self.data.loc[:,features])
        infoArr = np.array([[x,self.data.loc[x,self.class_id]] for x in range(self.data.shape[0])])
        nCluster = self.data.loc[:,[self.class_id]].nunique()[0]

        ftrArr = np.array(ftrArr)
        infoArr = np.array(infoArr)

        ftrArrWhite = whiten(ftrArr)
        centroids, distortion = kmeans(ftrArrWhite, nCluster)
        clusResults = -1*np.ones(ftrArrWhite.shape[0])

        for ii in range(ftrArrWhite.shape[0]):
            diff = centroids - ftrArrWhite[ii,:]
            diff = np.sum(np.power(diff,2), axis = 1)
            indMin = np.argmin(diff)
            clusResults[ii] = indMin

        ClusterOut = []
        classCluster = []
        globalDecisions = []
        for ii in range(nCluster):
            ind = np.where(clusResults==ii)[0]
            freqCnt = []
            for elem in infoArr[ind,1]:
                freqCnt.append(infoArr[ind,1].tolist().count(elem))
            indMax = np.argmax(freqCnt)
            classCluster.append(infoArr[ind,1][indMax])

            print("\n(Cluster: " + str(ii) + ") Using majority voting as a criterion this cluster belongs to " +
                  "class: " + str(classCluster[-1]))
            print ("Number of sounds in this cluster are: " + str(len(ind)))
            decisions = []
            for jj in ind:
                if infoArr[jj,1] == classCluster[-1]:
                    decisions.append(1)
                else:
                    decisions.append(0)
            globalDecisions.extend(decisions)
            print ("sound-id, sound-class, classification decision")
            ClusterOut.append(np.hstack((infoArr[ind],np.array([decisions]).T)))
            print (ClusterOut[-1])
        globalDecisions = np.array(globalDecisions)
        totalSounds = len(globalDecisions)
        nIncorrectClassified = len(np.where(globalDecisions==0)[0])
        return([ClusterOut, classCluster, centroids])

    #this plots a group of featurs against themselves.
    def scatter_plot(self, features, directory):
        colors = ['r', 'g', 'c', 'b', 'k', 'm', 'y']
        plt.figure(figsize= (10,10))
        for ii in range(len(features)):
            for jj in range(len(features)):
                plt.subplot(len(features),len(features), ii*len(features) + jj + 1)
                for i in range(self.data.shape[0]):
                    x_cord = self.data.loc[i,features[jj]]
                    y_cord = self.data.loc[i,features[ii]]
                    plt.scatter(x_cord,y_cord, c = colors[self.data.loc[i,self.class_id]], s=20, alpha=0.75)
                    #circ = Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor=colors[])
                    #legArray.append(circ)
                    if jj == 0:
                        plt.ylabel(features[ii])
                    else:
                        plt.yticks([])
                    if ii == len(features) -1:
                        plt.xlabel(features[jj])
                    else:
                        plt.xticks([])
        plt.savefig(directory)

#currently, this doesn't return anything regarding kmeans.
#It's not clear what it should return - the vote?
#the centroids?
#the distances from the centroid of each point?
#let me know and I'll bake it in.
#kmeans as a method of the class Analysis works, however.
def run_analysis(file_in = '/Users/samuelzetumer/Desktop/tabla-master/features/features.csv',
                path_out = '/Users/samuelzetumer/Desktop/tabla-master/analysis/',
                class_id = 'lung_disease',
                features = freq_features,
                n_comp_pca = 4,
                k_neighbors = 5,
                visual_features = visual_sample,
                visual_file_1 = 'viz1.png'):

    df = pd.read_csv(file_in)
    a1 = Analysis(data = df, class_id = class_id)

    a1.add_pca(features = features, n_comp = n_comp_pca)

    a1.train_knn(features = freq_features, k_neighbors = k_neighbors)

    #pca
    reduced = a1.models['pca'].fit_transform(df.loc[:,features])
    reduced_cols = ['pc{0}_value'.format(str(i)) for i in range(reduced.shape[1])]
    reduced_df = pd.DataFrame(reduced,columns = reduced_cols)

    #let's also get the coordinates of the first principal component:
    pc1 = a1.models['pca'].components_[0]
    pc_cols = ['{0}_pc1'.format(str(i)) for i in list(features)]
    pc_df = pd.DataFrame(pc1).transpose()
    pc_df.columns = pc_cols

    #knearest neighbors on original features
    knn_results = a1.models['knn'].predict_proba(df.loc[:,features])
    knn_cols = ['{0}_knn_vote'.format(str(i)) for i in df.loc[:,class_id].unique()]
    knn_df = pd.DataFrame(knn_results, columns = knn_cols)
    knn_score = a1.models['knn'].score(df.loc[:,features],df.loc[:,class_id])

    #now knn on pca'ed coordinates
    df2 = pd.concat([pd.DataFrame(reduced_df), df.loc[:,class_id]], axis=1)
    a2 = Analysis(data = df2, class_id = class_id)
    a2.train_knn(features = reduced_cols, k_neighbors = k_neighbors)

    knn2_results = a2.models['knn'].predict_proba(pd.DataFrame(reduced_df))
    knn2_cols = ['pc_{0}_knn_vote'.format(str(i)) for i in df.loc[:,class_id].unique()]
    knn2_df = pd.DataFrame(knn2_results, columns = knn2_cols)
    knn2_score = a1.models['knn'].score(reduced_df,df.loc[:,class_id])

    #now let's do kmeans
    #insert kmeans here.

    #single variate tests
    #ttests
    ttest_df = a1.single_variate_test(f_ttest, name = "ttest_")

    #now we have to construct the raw output of the transformation:
    transformations = pd.concat([reduced_df,knn_df,knn2_df], axis = 1)

    results_a = pd.DataFrame([knn_score] + [knn2_score])
    scrap = pd.DataFrame(results_a).transpose()
    result = pd.concat([scrap, ttest_df, pc_df], axis = 1).transpose()
    result2 = result.rename(index={0:'knn_score', 1:'knn2_score'})

        #destinations!
        #the results file is a singe
    results_path = path_out + "results.csv"
    transformations_path = path_out + "transformations.csv"
    visualize_path = path_out + visual_file_1
    visualize2_path = path_out + 'pca_viz.png'

    result2.to_csv(results_path)
    transformations.to_csv(transformations_path)

    #now it needs to plot some stuff.
    a1.scatter_plot(visual_features, visualize_path)
    a2.scatter_plot(reduced_cols, visualize2_path)

def f_ttest(col1, col2):
    result = sci.stats.ttest_ind(col1, col2, equal_var = False)
    return(result[1])
