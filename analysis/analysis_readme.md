#### run_analysis() creates the following files:

#### results.csv

1. knn\<2\>\_score
	* knn_score is the accuracy of k nearest neighbors on the features.
	* knn2_score is the accuracy of k nearest neighbors of the principal components of the features.
	
2. ttest_\<feature\>
	* this is the p-value of a t-test on this feature. The groups are determined by class_id values, default class_id is 'lung_disease'.
	
3. \<feature\>\_pc1:
	* this is a measure of the “importance” of that feature in the first principal component.
		* NOTE: this assumes that the frequency features have been normalized.
		* If the frequency features are not normalized, these feature_pc1 values will be influenced by the scale of each feature.

#### transformations.csv

1. pc\<number\>\_value:
	* these are the values of the principle components of each of the patient’s features.
2. knn\<2\>\_vote:
	* these are the number of each class of neighbor.
	* it’s possible that these columns are labelled incorrectly. It’s unclear how knn from scikit chooses which label is first. I assume numeric order, but it may do it by index.

##### viz1.png

This graphic depicts the distribution of the class_id classes, as plotted against the features provided by visual_features.
