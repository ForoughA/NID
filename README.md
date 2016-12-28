# Spectral Methods for Correlated Topic Models

This folder contains two subfolders below:

	* data/
	    This folders contains the Bag of Words dataset of New York Times articles available on [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/) . The data has been split to two sets of train and test and saved as a sparse MATLAB matrix (stored as “*.mat” files).

	* code/
	    This folder contains the [“spectral methods for correlated topic models”](https://arxiv.org/abs/1605.09080) code. The main file is called NIDtmMain.m. Once ran, the m-file will generate two variable called “perpNID”  and “PMI” in the workspace that are the resulting perplexity and PMI, respectively, for the dataset available in the data/ folder. Please refer to NIDtmMain.m for further documentation. The input can either be MATLAB’s *.mat or *.txt as explained in the NIDtmMain.m.

