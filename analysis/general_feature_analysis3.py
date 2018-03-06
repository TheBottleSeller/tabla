import numpy as np
import scipy as sci
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


#What follows is a set of parameters for figue output.
#'c': what color do you want the dots that belong to class 1, class 2, etc. to be.
#'s': what shape do you want the dots that belong to class 1, class 2, etc. to have.
#'title': the title of the graph, 'title_font_size', and
#'title_height' are self explanatory.
#'point_size': how big do you want the points to be? If it is set at zero, it will make the points smaller the more graphs that you have.
#'font_size': same as point size. Font size of x,y labels.
#label_coordinates: pretty self explanatory.
#'dimensions': make the actual picture bigger or smaller.
#'legend': currently not yet functional.
#'box': do you want the graph to be in a box, or do you want only the axes to have lines?
g1_params_default = {
    'c':['r', 'g', 'c', 'b', 'k', 'm', 'y'],
    'marker':['.', '.', '.', '.', '.', '.', '.'],
    'title' : 'Sample title',
    'title_font_size' : 24,
    'title_height' : .95,
    'point_size' : 0,
    'font_size' : 0,
    'x_label' : 'default',
    'y_label' : 'default',
    'x_label_y_coord' : 0,
    'y_label_x_coord' : 0,
    'dimensions' : (10,10),
    'legend' : 'xx',
    'box' : False
    }


#this set of parameters makes a bunch of points
#of just x's and o's
g1_params_paper = {
    'c' : ['k', 'k', 'k', 'k', 'k', 'k', 'k'],
    'marker': ['o', 'x', 'o', 'x', 'o', 'x', 'o'],
    'title' : 'Sample title',
    'title_font_size' : 24,
    'title_height' : .95,
    'point_size' : 0,
    'font_size' : 0,
    'x_label' : 'default',
    'y_label' : 'default',
    'x_label_y_coord' : 0,
    'y_label_x_coord' : 0,
    'dimensions' : (10,10),
    'legend' : 'xx',
    'box' : False
    }

#####################################################################

#This is an Analysis class that
#supports further statistical analyses for any data frame.
#enter the data as a dataframe.
#build models and put them in Analysis.models dictionary

class Analysis():

    def __init__(self, data, class_id, models ={}):
        self.data = data
        self.class_id = class_id
        self.models = models

    #this neatly applies a univariate analysis to each feature.
    #currently, only two classes of result feature are supported.
    #you can run a for loop for each class if you want to support
    #more classes.

    def single_variate_test(self, func, name):

        #group0
        df0 = self.data[self.data[self.class_id] == 0]

        #group1
        df1 = self.data[self.data[self.class_id] == 1]

        result_df = pd.DataFrame()

        #for each column in the data frame
        for col in list(self.data):
          #apply the statistical test "func"
          #to group0 and group1
            try:
                result = func(df0[col], df1[col])
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

    #plots a group of features against themselves.
    def scatter_plot(self,
                     features,
                     directory,
                     parameters):

        #sent over
        if parameters['font_size'] == 0:
            font_size = 20/(len(features)**.5)
        else:
            font_size = parameters['font_size']

        if parameters['point_size'] == 0:
            point_size = 240/len(features)
        else:
            point_size = parameters['point_size']

        #this is the size of the png
        plt.figure(figsize = parameters['dimensions'])

        #for each subgraph (each pair of features)...
        for ii in range(len(features)):
            for jj in range(len(features)):
              #skip it if the row index is higher than the column index
                if jj >= ii:
                    continue

                #This line is complicated. It basically adjusts the
                #Dimensions of the plots so that the excluded plots don't
                #take up a bunch of white space.
                ax = plt.subplot(len(features) - 1, len(features) - 1,
                            ((ii - 1) * len(features) + jj + 2 - ii))


                #get rid of the right and upper lines if you want to.
                if parameters['box'] == False:
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)

                ax.tick_params(labelsize = 20/(len(features)**.5))

                #add each point
                #there's probably a way to do this with arrays and no
                #for loop. Oh well.
                for i in range(self.data.shape[0]):
                    x_cord = self.data.loc[i,features[jj]]
                    y_cord = self.data.loc[i,features[ii]]
                    ax.scatter(x_cord,y_cord,
                                c = parameters['c'][self.data.loc[i,self.class_id]],
                                marker = parameters['marker'][self.data.loc[i,self.class_id]],
                                s = 240/len(features), alpha=0.75)

                 #INSERT INDIVIDUAL SUBPLOT FEATURES HERE
                 
                #Add the tick marks and labels only for the graphs
                #that are on the edges
                if jj == 0:
                    if parameters['y_label'] == 'default':
                        y_label = features[ii]
                    else:
                        y_label = parameters['y_label']
                    plt.ylabel(y_label, fontsize = font_size, x = parameters['y_label_x_coord'])
                else:
                    plt.yticks([])
                if ii == len(features) - 1:
                    if parameters['x_label'] == 'default':
                        x_label = features[jj]
                    else:
                        x_label = parameters['x_label']
                    plt.xlabel(x_label, fontsize = font_size, y = parameters['x_label_y_coord'])
                else:
                    plt.xticks([])

        #I'm not sure where the legend stuff goes, but you might want to put it here.

        plt.suptitle(parameters['title'], fontsize = parameters['title_font_size'], y = parameters['title_height'])
        plt.savefig(directory, bbox_inches='tight')

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
                visual_file_1 = 'viz1.png',
                g1_parameters = g1_params_default,
                g2_parameters = g1_params_default):

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
    a1.scatter_plot(visual_features, visualize_path, parameters = g1_parameters)
    a2.scatter_plot(reduced_cols, visualize2_path, parameters = g2_parameters)

def f_ttest(col1, col2):
    result = sci.stats.ttest_ind(col1, col2, equal_var = False)
    return(result[1])
