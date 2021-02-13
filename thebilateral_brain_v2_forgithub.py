#######################################################
# Python program name	: thebilateral_brain_v2.py
#Description	: code for paper Brain intra vs interlateral correlation of subcortical structures
#Args           :                                                                                      
#Author       	: Jaime Gomez-Ramirez                                               
#Email         	: jd.gomezramirez@gmail.com 
# Code extracted from bilateral_brain.py, select only code strictly necessary for the paper
#REMEMBER to Activate Keras source ~/github/code/tensorflow/bin/activate
#pyenv install 3.7.0
#pyenv local 3.7.0
#python3 -V
# To use ipython3 debes unset esta var pq contien old version
#PYTHONPATH=/usr/local/lib/python2.7/site-packages
#unset PYTHONPATH
# $ipython3
# To use ipython2 /usr/local/bin/ipython2
#/Library/Frameworks/Python.framework/Versions/3.7/bin/ipython3
#pip install rfpimp. (only for python 3)
#######################################################
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, pdb
import datetime

np.random.seed(11)
# Set figures dir

def plot_bar_zs(b_inv_z,l_inv_z,r_inv_z):
	"""
	"""
	print('Plotting inverse Z transform...')
	fig, ax = plt.subplots()
	plt.figure(figsize=(6, 4))
	x = ['Bilateral', 'LH', 'RH']
	zs = [b_inv_z,l_inv_z,r_inv_z]
	x_pos = [i for i, _ in enumerate(x)]
	plt.bar(x_pos, zs, color='green')
	plt.xlabel("Correlation Matrix")
	plt.ylabel("Z inverse transform")
	plt.xticks(x_pos, x)
	#ax.set_ylim([0,1])
	plt.title("Intra and Interhemispheric aggregate statistical dependence")
	plt.tight_layout()
	fig_name = os.path.join(figures_dir, 'barZinv.png')
	plt.savefig(fig_name)
	print('Saved Eigenvalues curve at %s' %fig_name)

def make_shuffled_df(df):
	"""
	"""
	import itertools
	df_shuffled = df.copy()
	#rows = cm.index.tolist()
	cols = df.columns.tolist()
	for col in cols:
		#rho= np.corrcoef(healthy[pair[0]].sample(frac=1),healthy[pair[1]].sample(frac=1))
		#df_shuffled.at[pair[0],pair[1]] = rho

		df_shuffled[col]=df[col].sample(frac=1).reset_index(drop=True)
	return df_shuffled

def average_correlations(cm, label=None):
	""" agreagate correlatiuon 
	Input df standarized from it get the cm
	"""
	import itertools
	cm_z = cm.copy()

	print('%s -- The average correlation is %.3f' %(label,cm.mean().mean()))
	# Z transform
	rows = cm.index.tolist()
	cols = cm.columns.tolist()
	pairs=list(itertools.product(rows,cols))
	for pair in pairs:
		rho = cm.at[pair[0],pair[1]]
		if pair[0] == pair[1]:
			# if rgo 1 Z is inf
			r_z =rho
		else:
			r_z = np.arctanh(rho)
		cm_z.at[pair[0],pair[1]]=r_z
	
	z_mean = cm_z.mean().mean()
	print('%s --The mean.mean of the Z Transform ==%.4f' %(label,z_mean))
	inv_z_mean = np.tanh(z_mean)
	print('%s --The inverse transform ==%.4f' %(label, inv_z_mean))
	return cm_z

def minmax_standard_df(df):
	""" normalize: x-min/(max-min); standarize=x-mi/std
	OUT: minmax, standarize
	"""
	from sklearn import preprocessing
	names = df.columns
	#x = df.values
	# Normalize
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(df)
	minmax_df = pd.DataFrame(x_scaled, columns=names)
	# standarize
	scaler = preprocessing.StandardScaler()
	scaled_df = scaler.fit_transform(df)
	scaled_df = pd.DataFrame(scaled_df, columns=names)
	return minmax_df, scaled_df

def distance_between_matrices(m1,m2):
	"""Returns the matrix of all pair-wise distances
	"""
	from scipy.spatial import distance_matrix
	mat_dist = distance_matrix(m1, m2)
	return mat_dist

def print_maxmin_cm(cm, label=None):
	"""
	"""
	stack = cm.mask(np.triu(np.ones(cm.shape)).astype(bool)).stack()
	mini, maxi = stack.idxmin(), stack.idxmax()
	print('Max correlation pair L %s %s - %s' %(label, maxi[0],maxi[1]))
	print('Min correlation pair R %s %s - %s' %(label, mini[0],mini[1]))
	return mini, maxi

def eigenvalues_analysis_corrmatrix(cm, label):
	"""
	"""
	print('Getting eigenvalues and eigenvectors of corr matrix...\n')
	eig_vals, eig_vecs = np.linalg.eigh(cm)
	#eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

	print('Eigenvalues of %s matrix are: %s' %(label, str(eig_vals)))
	# print in reports
	out_f  = os.path.join(reports_dir, 'eig_' + label + '.txt')
	text_file = open(out_f, "a")
	text_file.write('Spectral analysis for %s \n' %label)
	text_file.write('Eigenvalues: \n%s'  %eig_vals)
	text_file.write('Eigenvectors: \n%s'  %eig_vecs)
	print('Plotting curve of variance captured by eigen values')
	tot = sum(eig_vals)
	var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
	cum_var_exp = np.cumsum(var_exp)
	print('==%s Tot eigvals==%.4f 2 first Cumsum==%s' %(label, tot,cum_var_exp[0:2]))
	def plot_eigen_histogram_curve(var_exp, cum_var_exp,label=None):
		"""https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
		"""
		with plt.style.context('seaborn-whitegrid'):
			fig, ax = plt.subplots()
			plt.figure(figsize=(6, 4))
			plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',label='individual explained variance')
			plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
			plt.ylabel('Explained variance ratio')
			plt.xlabel('Principal components')
			plt.legend(loc='best')
			# plot diagonal red line
			x1=[0,6]
			y1=[0,100]
			plt.plot(x1,y1,'r--')
			plt.tight_layout()
			fig_name = os.path.join(figures_dir, label)
			plt.savefig(fig_name)
			print('Saved Eigenvalues curve at %s' %fig_name)
	plot_eigen_histogram_curve(var_exp, cum_var_exp, 'Eig_'+label+'.png')
	aoc = np.trapz(cum_var_exp)
	normaoc = str(aoc/700)
	text_file.write('AOC: \n%s'  %aoc)
	text_file.write('AOC normalized: \n%s'  %normaoc)
	text_file.close()
	return cum_var_exp

def get_corrmatrix(dataset, lateral, sex=None):
	""" return corr matrix based on condition
	"""
	R_cols = [s for s in dataset.columns if "_R_" in s]
	L_cols = [s for s in dataset.columns if "_L_" in s]

	if sex is not None:
		if sex == 'F':
			dataset =  dataset[dataset['sex']==1]
		elif sex == 'M':
			dataset =  dataset[dataset['sex']==0]
		else:
			print('ERROR sex must be F, M or None')
			retrurn -1
	corrmatrix = dataset.drop(['age','sex'], axis=1).corr(method='pearson')
	if lateral =='B':
		#selecting bilateral corr matrix columns
		corrmatrix = corrmatrix.loc[L_cols][R_cols]
	elif lateral == 'R':
		#selecting right side corr matrix columns
		corrmatrix = corrmatrix.loc[R_cols][R_cols]
	elif lateral == 'L':
		#selecting left side corr matrix columns
		corrmatrix = corrmatrix.loc[L_cols][L_cols]
	return corrmatrix

def plot_corrmatrix(cm, label):
	"""plot cm 
	"""
	# Plot correlations
	f = plt.figure(figsize=(19, 15))
	plt.matshow(cm, fignum=f.number)
	plt.xticks(range(cm.shape[1]), cm.columns, fontsize=14, rotation=45)
	plt.yticks(range(cm.shape[1]), cm.columns, fontsize=14)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=14)
	fig_name = os.path.join(figures_dir, label + '_corr_long.png')
	plt.savefig(fig_name)

def plot_boxandcorr(df, label=None):
	"""Plot boxplot and correlations for pandas df
	"""
	
	subcortical = df.columns.to_list()
	print('Median volumes of %s \n'% label)
	sorted_nb = df[subcortical].median().sort_values()
	print(sorted_nb)

	f = plt.figure(figsize=(19, 15))
	# Plot volumes increasing order
	ax = df.boxplot(column=sorted_nb.index.values.tolist(), rot=45, fontsize=14)
	ax.set_title('Subcortical volume estimates ')
	ax.set_ylabel(r'Volume in $mm^3$') #ax.set_xlabel(' ')
	fig_name = os.path.join(figures_dir, label + '_boxplot_long.png')
	plt.savefig(fig_name)
	# Plot correlations
	f = plt.figure(figsize=(19, 15))
	plt.matshow(df.corr(method='pearson'), fignum=f.number)
	plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
	plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=14)
	fig_name = os.path.join(figures_dir, label + '_corr_long.png')
	plt.savefig(fig_name)

def frequentist_tests_by_group(dfg1, dfg2):
	from scipy.stats import ttest_ind
	df = pd.DataFrame(columns=['roi', 'ttest', 'pval'])
	for i in range(len(dfg1.columns)):
		col = dfg1.columns[i] #==dfg2 have same columns
		y1, y2 = dfg1[col], dfg2[col]
		ttest,pval = ttest_ind(y1,y2)
		new_row = {'roi':col, 'ttest':ttest, 'pval':pval}
		df = df.append(new_row, ignore_index=True)	
		print('\t col =%s %.5f' %(col, pval))
	return df

def frequentist_tests_by_col(dataset, col1, col2, label=None):
	from scipy.stats import ttest_ind
	y1 = dataset[col1]
	y2 = dataset[col2]
	ttest,pval = ttest_ind(y1,y2)
	return ttest,pval

def plot_removed_foriso(indexes, fsl_counts, free_counts, label=None):
	""" Plot stacked bar #removed/remain based on isoforest contamination param
	"""
	df = pd.DataFrame({'contamination': indexes, 'fsl removed':fsl_counts, 'free removed':free_counts}, index=indexes)
	ax = df.plot.bar(rot=0)
	ax.set_xlabel('IsoForest Contamination')
	ax.set_ylabel('$\\%$ removed cases')
	fig_name = os.path.join(figures_dir, 'PCremovedIsoforest_' + str(label) + '.png')
	plt.savefig(fig_name)
	return df

def outlier_detection_isoforest(X, contamination, y=None):
	"""https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
	"""
	from sklearn.ensemble import IsolationForest

	iso = IsolationForest(random_state=0, contamination=contamination)
	print(iso)
	yhat = iso.fit_predict(X)
	# select all rows that are not outliers
	mask = yhat != -1
	print('Number of outliers (rows) removed = %.d / %.d' %(sum(mask==False), yhat.shape[0]))
	if y is None:
		return  X[mask]
	else:
		X_train, y_train = X[mask], y[mask]
		return X_train, y_train

def main():
	"""
	"""
	plt.close('all')
	# Open csv dataset
	dataframe = pd.read_csv(csv_path, sep=';') 
	dataframe_orig = dataframe.copy()
	# FLS Free col names of subcortical structures
	fsl_cols = ['R_Thal_visita1','L_Puta_visita1','L_Amyg_visita1','R_Pall_visita1','L_Caud_visita1',\
	'L_Hipp_visita1','R_Hipp_visita1','L_Accu_visita1','R_Puta_visita1','BrStem_visita1',\
	'R_Caud_visita1','R_Amyg_visita1','L_Thal_visita1','L_Pall_visita1','R_Accu_visita1']
	free_cols = ['fr_Right_Thalamus_Proper_y1','fr_Left_Putamen_y1','fr_Left_Amygdala_y1',\
	'fr_Right_Pallidum_y1','fr_Left_Caudate_y1','fr_Left_Hippocampus_y1','fr_Right_Hippocampus_y1',\
	'fr_Left_Accumbens_area_y1','fr_Right_Putamen_y1','fr_Right_Caudate_y1','fr_Right_Amygdala_y1',\
	'fr_Left_Thalamus_Proper_y1','fr_Left_Pallidum_y1','fr_Right_Accumbens_area_y1']
	# Rename for longitudinal df to be created
	fsl_lon_cols = ['fsl_R_Thal', 'fsl_L_Thal', 'fsl_R_Puta', 'fsl_L_Puta','fsl_R_Amyg', 'fsl_L_Amyg', 'fsl_R_Pall', 'fsl_L_Pall', 'fsl_R_Caud','fsl_L_Caud', 'fsl_R_Hipp', 'fsl_L_Hipp', 'fsl_R_Accu', 'fsl_L_Accu']
	free_lon_cols = ['free_R_Thal', 'free_L_Thal', 'free_R_Puta','free_L_Puta', 'free_R_Amyg', 'free_L_Amyg', 'free_R_Pall','free_L_Pall', 'free_R_Caud', 'free_L_Caud', 'free_R_Hipp','free_L_Hipp', 'free_R_Accu', 'free_L_Accu']
	# Rename for RH and LH
	fsl_R_cols = [s for s in fsl_lon_cols if "_R_" in s]
	fsl_L_cols = [s for s in fsl_lon_cols if "_L_" in s]
	free_R_cols = [s for s in free_lon_cols if "_R_" in s]
	free_L_cols = [s for s in free_lon_cols if "_L_" in s]

	df_fsl_lon_all = df_fsl_lon.copy()
	df_free_lon_all = df_free_lon.copy()

	# analysis conditions select first tool 
	tool = ['free', 'fsl']
	ix_tool = 0
	tool = tool[ix_tool]
	if tool =='fsl':
		df_lon= df_fsl_lon
		cols_lon = fsl_lon_cols
	elif tool == 'free':
		df_lon= df_free_lon
		cols_lon = free_lon_cols
	
	# Delete contamination on entire DF	
	fsl_totalr = df_fsl_lon.shape[0]
	free_totalr = df_free_lon.shape[0]
	contamination = [0.01, 0.05, 0.1, 'auto']
	ix_contlabel = -1
	cont_label = str(contamination[ix_contlabel])
	fsl_datasets, free_datasets, indexes, fsl_counts, free_counts = [], [], [], [],[]
	for cont in contamination:
		print('outlier_detection_isoforest contamination =%s \n' %cont)
		df_fsl_lon_post = outlier_detection_isoforest(df_fsl_lon, cont)
		df_free_lon_post = outlier_detection_isoforest(df_free_lon, cont)
		print('Sick PRE Iso %d POST Iso = %d' %(sum(df_free_lon['dx_visita']>0),sum(df_free_lon_post['dx_visita']>0)))
		print('Removed healthys = %d Sicks ==%d' %(sum(df_free_lon['dx_visita']==0) - sum(df_free_lon_post['dx_visita']==0), sum(df_free_lon['dx_visita']>0) - sum(df_free_lon_post['dx_visita']>0)))

		# percentange of remaining cases
		fsl_removed_pc = 1 - (df_fsl_lon_post.shape[0] - fsl_totalr)/100
		free_removed_pc =1 - (df_free_lon_post.shape[0] - free_totalr)/100
		fsl_datasets.append(df_fsl_lon_post), free_datasets.append(df_free_lon_post)
		indexes.append(cont), fsl_counts.append(fsl_removed_pc), free_counts.append(free_removed_pc)

	df_plot = plot_removed_foriso(indexes, fsl_counts, free_counts)
	print('Bar plot removed_foriso at PCremovedIsoforest_ISO\n')

	# Longitudinal with contamination filter
	# Select auto contamination -1 for auto
	df_fsl_lon, df_free_lon = fsl_datasets[ix_contlabel], free_datasets[ix_contlabel]
	## EDA  box plot and correlation matrix
	print('Box and Corr plots for ALL FS and FSL...\n')
	# All regions and conditions
	label = 'free' + cont_label + '_ALL'
	plot_boxandcorr(df_free_lon[free_lon_cols], label)
	label = 'fsl' + cont_label + '_ALL'
	plot_boxandcorr(df_fsl_lon[fsl_lon_cols], label)

	# Select uncontaminated df based on condition
	# Select based on DX
	# Only Healthy
	df_free_H_lon = df_free_lon.loc[df_free_lon['dx_visita']==0]
	df_free_H_lon = df_free_H_lon.reset_index(drop=True)
	df_fsl_H_lon = df_fsl_lon.loc[df_fsl_lon['dx_visita']==0]
	df_fsl_H_lon = df_fsl_H_lon.reset_index(drop=True)

	# Non AD (H or MCI)
	df_free_NAD_lon = df_free_lon.loc[df_free_lon['dx_visita'].isin([0,1])]
	df_free_NAD_lon = df_free_NAD_lon.reset_index(drop=True)
	df_fsl_NAD_lon = df_fsl_lon.loc[df_fsl_lon['dx_visita'].isin([0,1])]
	df_fsl_NAD_lon = df_fsl_NAD_lon.reset_index(drop=True)
	# Only MCIs
	df_free_MCI_lon = df_free_lon.loc[df_free_lon['dx_visita']==1]
	df_free_MCI_lon = df_free_MCI_lon.reset_index(drop=True)
	df_fsl_MCI_lon = df_fsl_lon.loc[df_fsl_lon['dx_visita']==1]
	df_fsl_MCI_lon = df_fsl_MCI_lon.reset_index(drop=True)
	# Only sick ones
	df_free_AD_lon = df_free_lon.loc[df_free_lon['dx_visita']==2]
	df_free_AD_lon = df_free_AD_lon.reset_index(drop=True)
	df_fsl_AD_lon = df_fsl_lon.loc[df_fsl_lon['dx_visita']==2]
	df_fsl_AD_lon = df_fsl_AD_lon.reset_index(drop=True)

	####################################################################
	## H vs SICK
	print('\n\n ---- ANALYSIS Condition based tool==%s ---\n\n' %tool)	
	# test based on diagnostics
	label = tool+ '_DXH-AD'
	if tool =='free':
		healthy = df_free_H_lon[free_lon_cols]
		ad = df_free_AD_lon[free_lon_cols]
	elif tool=='fsl':
		healthy = df_fsl_H_lon[fsl_lon_cols]
		ad = df_fsl_AD_lon[fsl_lon_cols]
	# Normalize dataset
	healthy_mM, healthy_std = minmax_standard_df(healthy)
	######################################################################
	df_test_dx = frequentist_tests_by_group(healthy,ad)
	reportf= os.path.join(reports_dir, 'ttestby_' + str(label) + '.txt')
	df_test_dx.to_csv(reportf, header=None, index=None, sep=' ', mode='w')
	label = tool + cont_label + '_H'
	plot_boxandcorr(healthy, label)

	#######################################################################
	### Select DX condition
	# Select df lon for analysis
	status_label = ['ALL', 'H','MCI','AD','NAD']
	ix_statuslabel = 1 # Healthy and Condition
	status_label = status_label[ix_statuslabel]
	if status_label == 'H':
		df_fsl_lon = df_fsl_H_lon
		df_free_lon = df_free_H_lon
	elif status_label == 'MCI':
		df_fsl_lon = df_fsl_MCI_lon
		df_free_lon = df_free_MCI_lon
	elif status_label == 'AD':
		df_fsl_lon = df_fsl_AD_lon
		df_free_lon = df_free_AD_lon
	elif status_label == 'NAD':
		df_fsl_lon = df_fsl_NAD_lon
		df_free_lon = df_free_NAD_lon
	elif status_label == 'ALL':
		print('Analysis  for ALL no condition applied')

	# ConditionRX && Condition
	# test whether differences by groups: hand/sex/apoe
	# 1 diestro, 2 zurdo 3 ambi 4 z.contrariado
	label = tool + '_'+ status_label + '_hand'
	rights = df_lon[cols_lon].where(df_lon['handlat']==1.0).dropna()
	filter_left = (df_lon['handlat']==2.0) |  (df_lon['handlat']==4.0)
	lefts = df_lon[cols_lon].where(filter_left).dropna()
	ambis = df_lon[cols_lon].where(df_lon['handlat']==3.0).dropna()
	df_test_hand = frequentist_tests_by_group(rights,lefts)
	reportf= os.path.join(reports_dir, 'ttestby_' + str(label) + '.txt')
	df_test_hand.to_csv(reportf, header=None, index=None, sep=' ', mode='w')

	label = tool + '_'+ status_label + '_sex'
	males = df_lon[cols_lon].where(df_lon['sex']==0).dropna()
	females = df_lon[cols_lon].where(df_lon['sex']==1).dropna()
	df_test_sex = frequentist_tests_by_group(males,females)
	reportf= os.path.join(reports_dir, 'ttestby_' + str(label) + '.txt')
	df_test_sex.to_csv(reportf, header=None, index=None, sep=' ', mode='w')

	label = tool + '_'+ status_label + '_apoe'
	apoe0 = df_lon[cols_lon].where(df_lon['apoe']==0).dropna()
	apoe12 = df_lon[cols_lon].where(df_lon['apoe']>=1).dropna()
	df_test_apoe = frequentist_tests_by_group(apoe0,apoe12)
	reportf = os.path.join(reports_dir, 'ttestby_' + str(label) + '.txt')
	df_test_apoe.to_csv(reportf, header=None, index=None, sep=' ', mode='w')
	### EDA 
	# Plot by diagnostic
	label =  tool + '_DX_' + cont_label
	plot_boxandcorr(healthy, label+ '_H')
	plot_boxandcorr(ad, label+ '_AD')

	# Plot by Hand condition
	label =  tool + '_'+ status_label + cont_label
	plot_boxandcorr(rights, label+ '_Rights')
	plot_boxandcorr(lefts, label+ '_Lefts')
	plot_boxandcorr(ambis, label+ '_Ambis')
	# Plot by Sex condition
	plot_boxandcorr(males, label+ '_Males')
	plot_boxandcorr(females, label+ '_Females')
	# Plot by Apoe condition
	plot_boxandcorr(apoe0, label+ '_APOE0')
	plot_boxandcorr(apoe12, label+ '_APOE12')
	###### Eigenvalues Analysis
	####################################################################
	## Symmetry Study. Distance between hemispheres
	# Get correlation matrix by side
	print('Getting corrmatrices [Laterality B,L,M] [Sex F,M]')
	# FREE
	R_cols = [s for s in df_lon.columns if "_R_" in s]
	L_cols = [s for s in df_lon.columns if "_L_" in s]	
	# All conditions , esto no lo uso, borra
	dfsquared = df_lon[cols_lon]
	corrmatrix = dfsquared.corr(method='pearson')
	corrmatrix_B = corrmatrix.loc[L_cols][R_cols]
	corrmatrix_R = corrmatrix.loc[R_cols][R_cols]
	corrmatrix_L = corrmatrix.loc[L_cols][L_cols]
	
	# H vs SICK
	covariance = False # covariance or correlation

	healthy_shuffle = make_shuffled_df(healthy)
	print('Shuffled mean of CM ==%.4f' %healthy_shuffle.corr(method='pearson').mean().mean())
	#healthy.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)
	cm_h = healthy.corr(method='pearson')

	standarize, coda = True, ''
	if standarize == True:
		cm_h = healthy_std.corr(method='pearson')
		cm_z_allh=average_correlations(cm_h, 'ALL_H')
		coda = '_std'
	if covariance==True:
		cm_h = healthy.cov()
	cm_h_B = cm_h.loc[L_cols][R_cols]
	cm_h_R = cm_h.loc[R_cols][R_cols]
	cm_h_L = cm_h.loc[L_cols][L_cols]
	label = tool+cont_label+'H_B'+coda
	plot_corrmatrix(cm_h_B, label)
	cm_z_b=average_correlations(cm_h_B, label)

	label = tool+cont_label+'H_R'+coda
	plot_corrmatrix(cm_h_R, label)
	cm_z_r=average_correlations(cm_h_R, label)

	label = tool+cont_label+'H_L'+coda
	plot_corrmatrix(cm_h_L, label)
	cm_z_l=average_correlations(cm_h_L, label)
	#corrl = cm_z_l.corr().stack()
	#corr_l = corrl[corrl.index.get_level_values(0) != corrl.index.get_level_values(1)]
	masku=np.triu(cm_z_b,k=1)>0
	maski=np.tril(cm_z_b,k=-1)>0
	b_z_mean=(np.triu(cm_z_b,k=1)[masku].mean() + np.tril(cm_z_b,k=-1)[maski].mean())/2.
	b_inv_z = np.tanh(b_z_mean) 
	#L and R are symmetric just need triu or tril
	l_z_mean=np.triu(cm_z_l,k=1)[masku].mean()
	l_inv_z = np.tanh(l_z_mean) 
	r_z_mean=np.triu(cm_z_r,k=1)[masku].mean() #+ np.tril(cm_z_r,k=-1).mean()
	r_inv_z = np.tanh(r_z_mean) 
	print('***Inverse Z transform mean excluded Diag B==%.4f L==%.4f R==%.4f' %(b_inv_z,l_inv_z,r_inv_z))
	plot_bar_zs(b_inv_z,l_inv_z,r_inv_z)
	
	cm_ad = ad.corr(method='pearson')
	if covariance==True:
		cm_ad = ad.cov()
	cm_ad_B = cm_ad.loc[L_cols][R_cols]
	cm_ad_R = cm_ad.loc[R_cols][R_cols]
	cm_ad_L = cm_ad.loc[L_cols][L_cols]
	
	label2p = tool +'_DX_H'
	cum_var_exp2B = eigenvalues_analysis_corrmatrix(cm_h_B,label2p +'B'+coda)	
	cum_var_exp2R = eigenvalues_analysis_corrmatrix(cm_h_R,label2p +'R'+coda)
	cum_var_exp2L = eigenvalues_analysis_corrmatrix(cm_h_L,label2p +'L'+coda)
	norm =  cum_var_exp2B.shape[0]*100
	B_auc, L_auc, R_auc = np.trapz(cum_var_exp2B)/norm,np.trapz(cum_var_exp2L)/norm,np.trapz(cum_var_exp2R)/norm
	print('\n***\t\t%s FREE AUC B=%.4f L= %.4f R= %.4f' %(label2p, B_auc, L_auc, R_auc))
	pdb.set_trace()
	
	label2p = tool +'_DX_AD'
	if cm_ad_B.isnull().any().any():
		print('---Skipping AD DataFrame too few cases for Eigen Analysis')
	else:
		cum_var_exp2B = eigenvalues_analysis_corrmatrix(cm_ad_B,label2p +'B')	
		cum_var_exp2R = eigenvalues_analysis_corrmatrix(cm_ad_R,label2p +'R')
		cum_var_exp2L = eigenvalues_analysis_corrmatrix(cm_ad_L,label2p +'L')
		norm =  cum_var_exp2B.shape[0]*100
		B_auc, L_auc, R_auc = np.trapz(cum_var_exp2B)/norm,np.trapz(cum_var_exp2L)/norm,np.trapz(cum_var_exp2R)/norm
		print('\n***\t\t%s FREE AUC B=%.4f L= %.4f R= %.4f' %(label2p, B_auc, L_auc, R_auc))
	
	pdb.set_trace()
	# Sex condition
	cm_f = females.corr(method='pearson')
	if covariance==True:
		cm_f = females.cov()
	cm_f_B = cm_f.loc[L_cols][R_cols]
	cm_f_R = cm_f.loc[R_cols][R_cols]
	cm_f_L = cm_f.loc[L_cols][L_cols]
	cm_m = males.corr(method='pearson')
	if covariance==True:
		cm_m = males.cov()
	cm_m_B = cm_m.loc[L_cols][R_cols]
	cm_m_R = cm_m.loc[R_cols][R_cols]
	cm_m_L = cm_m.loc[L_cols][L_cols]
	# Hand laterality
	cm_r = rights.corr(method='pearson')
	if covariance==True:
		cm_r = rights.cov()
	cm_r_B = cm_r.loc[L_cols][R_cols]
	cm_r_R = cm_r.loc[R_cols][R_cols]
	cm_r_L = cm_r.loc[L_cols][L_cols]
	cm_l = lefts.corr(method='pearson')
	if covariance==True:
		cm_l = lefts.cov()
	cm_l_B = cm_l.loc[L_cols][R_cols]
	cm_l_R = cm_l.loc[R_cols][R_cols]
	cm_l_L = cm_l.loc[L_cols][L_cols]
	cm_ambi = ambis.corr(method='pearson')
	if covariance==True:
		cm_ambi = ambis.cov()
	cm_ambi_B = cm_ambi.loc[L_cols][R_cols]
	cm_ambi_R = cm_ambi.loc[R_cols][R_cols]
	cm_ambi_L = cm_ambi.loc[L_cols][L_cols]
	# APOE
	cm_a0 = apoe0.corr(method='pearson')
	if covariance==True:
		cm_a0 = apoe0.cov()
	cm_a0_B = cm_a0.loc[L_cols][R_cols]
	cm_a0_R = cm_a0.loc[R_cols][R_cols]
	cm_a0_L = cm_a0.loc[L_cols][L_cols]
	cm_a12 = apoe12.corr(method='pearson')
	if covariance==True:
		cm_a12 = apoe12.cov()
	cm_a12_B = cm_a12.loc[L_cols][R_cols]
	cm_a12_R = cm_a12.loc[R_cols][R_cols]
	cm_a12_L = cm_a12.loc[L_cols][L_cols]
	# print max min of cm
	output = print_maxmin_cm(cm_f_B, 'f_B')
	output = print_maxmin_cm(cm_f_R, 'f_R')
	output = print_maxmin_cm(cm_f_L, 'f_L')
	# Eigenvalue analysis


	label2p = tool + '_'+status_label+'_HandRight_'
	cum_var_exp2B = eigenvalues_analysis_corrmatrix(cm_ambi_B,label2p +'B')	
	cum_var_exp2R = eigenvalues_analysis_corrmatrix(cm_ambi_R,label2p +'R')
	cum_var_exp2L = eigenvalues_analysis_corrmatrix(cm_ambi_L,label2p +'L')
	norm =  cum_var_exp2B.shape[0]*100
	B_auc, L_auc, R_auc = np.trapz(cum_var_exp2B)/norm,np.trapz(cum_var_exp2L)/norm,np.trapz(cum_var_exp2R)/norm
	print('\n***\t\t%s FREE AUC B=%.4f L= %.4f R= %.4f' %(label2p, B_auc, L_auc, R_auc))
	label2p = tool + '_'+ status_label+'_HandLeft_'
	cum_var_exp2B = eigenvalues_analysis_corrmatrix(cm_l_B,label2p +'B')	
	cum_var_exp2R = eigenvalues_analysis_corrmatrix(cm_l_R,label2p +'R')
	cum_var_exp2L = eigenvalues_analysis_corrmatrix(cm_l_L,label2p +'L')
	B_auc, L_auc, R_auc = np.trapz(cum_var_exp2B)/norm,np.trapz(cum_var_exp2L)/norm,np.trapz(cum_var_exp2R)/norm
	print('\n***\t\t%s FREE AUC B=%.4f L= %.4f R= %.4f' %(label2p, B_auc, L_auc, R_auc))
	label2p = tool + '_'+status_label+'_HandAmbi_'
	cum_var_exp2B = eigenvalues_analysis_corrmatrix(cm_l_B,label2p +'B')	
	cum_var_exp2R = eigenvalues_analysis_corrmatrix(cm_l_R,label2p +'R')
	cum_var_exp2L = eigenvalues_analysis_corrmatrix(cm_l_L,label2p +'L')
	B_auc, L_auc, R_auc = np.trapz(cum_var_exp2B)/norm,np.trapz(cum_var_exp2L)/norm,np.trapz(cum_var_exp2R)/norm
	print('\n***\t\t%s FREE AUC B=%.4f L= %.4f R= %.4f' %(label2p, B_auc, L_auc, R_auc))
	
	label2p = tool + '_'+status_label+'_SexMale_'
	cum_var_exp2B = eigenvalues_analysis_corrmatrix(cm_m_B,label2p +'B')	
	cum_var_exp2R = eigenvalues_analysis_corrmatrix(cm_m_L,label2p +'R')
	cum_var_exp2L = eigenvalues_analysis_corrmatrix(cm_m_R,label2p +'L')
	norm =  cum_var_exp2B.shape[0]*100
	B_auc, L_auc, R_auc = np.trapz(cum_var_exp2B)/norm,np.trapz(cum_var_exp2L)/norm,np.trapz(cum_var_exp2R)/norm
	print('\n***\t\t%s FREE AUC B=%.4f L= %.4f R= %.4f' %(label2p, B_auc, L_auc, R_auc))
	label2p = tool + '_'+status_label+'_SexFemale_'
	cum_var_exp2B = eigenvalues_analysis_corrmatrix(cm_f_B,label2p +'B')	
	cum_var_exp2R = eigenvalues_analysis_corrmatrix(cm_f_R,label2p +'R')
	cum_var_exp2L = eigenvalues_analysis_corrmatrix(cm_f_L,label2p +'L')
	B_auc, L_auc, R_auc = np.trapz(cum_var_exp2B)/norm,np.trapz(cum_var_exp2L)/norm,np.trapz(cum_var_exp2R)/norm
	print('\n***\t\t%s FREE AUC B=%.4f L= %.4f R= %.4f' %(label2p, B_auc, L_auc, R_auc))
	
	# compute distance between matrices
	# B,L,R in the H group
	dlr= distance_between_matrices(cm_h_L,cm_h_R)
	dlb= distance_between_matrices(cm_h_L,cm_h_B)
	drb = distance_between_matrices(cm_h_R,cm_h_B)
	print('Distance Healthy LH-RH =%.2f LH-B =%.2f RH-B =%.2f' %(dlr.mean(), dlb.mean(),drb.mean()))
	# B,L,R distance in Females and in Male groups	
	dlr= distance_between_matrices(cm_f_L,cm_f_R)
	dlb= distance_between_matrices(cm_f_L,cm_f_B)
	drb = distance_between_matrices(cm_f_R,cm_f_B)
	print('Distance H Females LH-RH =%.2f LH-B =%.2f RH-B =%.2f' %(dlr.mean(), dlb.mean(),drb.mean()))
	dlr= distance_between_matrices(cm_m_L,cm_m_R)
	dlb= distance_between_matrices(cm_m_L,cm_m_B)
	drb = distance_between_matrices(cm_m_R,cm_m_B)
	print('Distance H Males LH-RH =%.2f LH-B =%.2f RH-B =%.2f' %(dlr.mean(), dlb.mean(),drb.mean()))
	pdb.set_trace()
	
	
	
	

	
	
	
	

	
	
	
	
