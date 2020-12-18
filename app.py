import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from settings import DATASET_DIR as dataset
from settings import IMAGE_DIR 
from sklearn import metrics
import pylab as pl
import scipy.optimize as opt
from sklearn.model_selection import train_test_split




def main():
	cell_df = pd.read_csv('dataset/cell_samples.csv')
	num = 6
	st.title("Support Vector Machine (SVM) Implementation")
	st.write("SVM works by mapping data to a high-dimensional feature space so that data points can \
		be categorized, even when the data are not otherwise linearly separable. A separator between\
		 the categories is found, then the data is transformed in such a way that the separator could \
		 be drawn as a hyperplane.")

	st.image('img/svm1.jpg')

	st.sidebar.title("Evaluating different parameters")
	st.sidebar.subheader("View dataset")
	num = st.sidebar.number_input("Choose number of data to view", 5, 30)
	if st.sidebar.checkbox('Show data'):
		st.write(cell_df.head(num))
		st.write("Total size of data: ", cell_df.shape)

	val = cell_df['Class'].value_counts().to_frame()


	# st.subheader("Visualizing data")
	# plt.figure(figsize=(5,3))
	# plt.hist(cell_df['Class'], bins=20, rwidth=0.9)
	# plt.grid(axis='y', alpha=0.75)
	# plt.xlabel('Class')
	# plt.ylabel('Counts')
	# plt.title('Benign(class=2) / Malignant (class=4)')
	# st.pyplot()

	#Distribution of classes based on Clump Thickness and Uniformity of cell size
	if st.sidebar.checkbox("Scatter of plot"):
		ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
		st.pyplot()
	
	#show the data types for each column
	if st.sidebar.checkbox("View datatype"):
		st.subheader("Data type of each column")
		#dropping values that are not non-numerical
		cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
		cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
		st.write(cell_df.dtypes)

	##Data pre-processing and selection
	cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
	cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')


	feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
	X = np.asarray(feature_df)

	cell_df['Class'] = cell_df['Class'].astype('int')
	y = np.asarray(cell_df['Class'])

	#Train/test data split
	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
	if st.sidebar.checkbox("Show number of train / test data"):
		st.write('Number of train set:', X_train.shape,  y_train.shape)
		st.write('Number of test set:', X_test.shape,  y_test.shape)


	#Modeling(SVM with Sci-Kit learn)
	st.sidebar.subheader("Select kernel")
	kernel_option = st.sidebar.selectbox('Kernel type',('Choose a kernel', 'Linear', 'Polynomial', 'Radial basis function (RBF)', 'Sigmoid'))
	from sklearn import svm
	if kernel_option == 'Radial basis function (RBF)':
		clf = svm.SVC(kernel='rbf')
		clf.fit(X_train, y_train)
		yhat = clf.predict(X_test)
		st.write("On using Radial basis function (RBF)")

	if kernel_option == 'Linear':
		clf = svm.SVC(kernel='linear')
		clf.fit(X_train, y_train)
		yhat = clf.predict(X_test)
		st.write("On using Linear")

	if kernel_option == 'Polynomial':
		clf = svm.SVC(kernel='poly')
		clf.fit(X_train, y_train)
		yhat = clf.predict(X_test)
		st.write("On using Polynomial")

	if kernel_option == 'Sigmoid':
		clf = svm.SVC(kernel='sigmoid')
		clf.fit(X_train, y_train)
		yhat = clf.predict(X_test)
		st.write("On using Sigmoid")



	##Accuracy 
	from sklearn.metrics import f1_score#jaccard_similarity_score,
	st.sidebar.subheader("Check accuracy")
	acc = st.sidebar.selectbox('Method',('Choose a method', 'F1 Score', 'Jaccard accuracy'))
	if acc == 'F1 Score':
		st.write("F1 Score: ", f1_score(y_test, yhat, average='weighted'))

	if acc == 'Jaccard accuracy':
		from sklearn.metrics import jaccard_score
		st.write("Jaccard accuracy: ", jaccard_score(y_test, yhat, pos_label=4))

	

	st.sidebar.write("\n")




if __name__=='__main__':
	main()


