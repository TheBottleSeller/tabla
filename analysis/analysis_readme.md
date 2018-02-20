#####RUN ANALYSIS####

run_analysis() creates the following files.

results.csv, which contains:
	a) the score (accuracy) of each of the knns:
		1) the first knn_score is the accuracy of k nearest neighbors on the features.
		2) knn2_score is the accuracy of k nearest neighbors of the principal components of the features.
	b) ttest_features:
		1) this is the p-value of a t-test on this variable. The groups are partitioned according
to the “class_id”, which in this case is ‘lung_disease’.
	c) features_pc1:
		1) this is a measure of the “importance” of that variable in the first principle component.
			i) NOTE: this assumes that the frequency features have been normalized.
			ii) If the frequency features are not normalized, these feature_pc1 values
will be influenced by the scale of each feature.

transformations.csv, which contains:
	a) pc<number>_value:
		1) these are the values of the principle components of each of the patient’s features.
	b) knn(2)_vote:
		2) these are the number of each class of neighbor.
			ii) it’s possible that these columns are labelled incorrectly. It’s unclear how knn from scikit chooses which label is first. I assume numeric order, but it may do it by index. I can fix that later.
