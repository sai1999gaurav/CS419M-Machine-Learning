import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
np.set_printoptions(suppress=True) 

#GLOBAL VARIABLES
feature = 5
neta = 0.25 #learning rate
epsilon = 1e-5
lamda = 1e-5

""" 
You are allowed to change the names of function arguments as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""
def get_phi(file_path):
	with open(file_path, 'r') as f:
		reader = csv.reader(f)
		next(reader)
		x = list(reader)
	row_length = int(len(x))
	phi = []
	for row in x:
		#datetime = sr.dt.strptime('% B % d, % Y, % r')
		dtime = row[0].split(':')
		dtime1 = dtime[0].split('-')
		year = dtime1[0]
		month = dtime1[1]
		day = dtime1[2].split()
		hour = day[1]
		day = day[0]
		## NORMALISATION
		hour = float(hour)/24.0
		year = (float(year)-2009)/6.0
		#print(hour)
		#latitude 102 - 109, long 62 - 72, pass <=6
		src_lat = (float(row[1]) + 109)/7.0
		src_long = (float(row[2]) -62)/10.0
		dest_lat = (float(row[3]) + 109)/7.0
		dest_long = (float(row[4]) -62)/10.0
		n_pass = float(row[5])/6.0
		dist = ((dest_lat-src_lat)**2 + (dest_long-src_long)**2)**0.5
		phi.append([1, year, hour, dist, n_pass])
		#np.append(([1, hour, src_lat, src_long, dest_lat, dest_long, n_pass]))
	#print(phi)
	#print(y)
	phi = np.array(phi)
	return phi

def get_features(file_path):
	# Given a file path , return feature matrix and target labels 
	phi = get_phi(file_path)
	with open(file_path, 'r') as f:
		reader = csv.reader(f)
		next(reader)
		x = list(reader)
	row_length = int(len(x))
	y = []
	for row in x:
		#datetime = sr.dt.strptime('% B % d, % Y, % r')
		fare = float(row[6])/170
		y.append(fare)	
		#np.append(([1, hour, src_lat, src_long, dest_lat, dest_long, n_pass]))
	#print(phi)
	#print(y)
	y = np.array(y)
	return phi, y


def get_phi_basis_1(file_path):
	with open(file_path, 'r') as f:
		reader = csv.reader(f)
		next(reader)
		x = list(reader)
	row_length = int(len(x))
	phi = []
	for row in x:
		#datetime = sr.dt.strptime('% B % d, % Y, % r')
		dtime = row[0].split(':')
		dtime1 = dtime[0].split('-')
		year = dtime1[0]
		month = dtime1[1]
		day = dtime1[2].split()
		hour = day[1]
		day = day[0]
		## NORMALISATION
		hour = (float(hour)/24.0)**0.5 #changed here
		year = ((float(year)-2009)/6.0)**0.5
		#print(hour)
		#latitude 102 - 109, long 62 - 72, pass <=6
		src_lat = (float(row[1]) + 109)/7.0
		src_long = (float(row[2]) -62)/10.0
		dest_lat = (float(row[3]) + 109)/7.0
		dest_long = (float(row[4]) -62)/10.0
		n_pass = (float(row[5])/6.0)**2  #CHANGED HERE
		dist = ((dest_lat-src_lat)**2 + (dest_long-src_long)**2)**0.5
		phi.append([1, year, hour, dist, n_pass])
		#np.append(([1, hour, src_lat, src_long, dest_lat, dest_long, n_pass]))
	#print(phi)
	#print(y)
	phi = np.array(phi)
	return phi

def get_features_basis1(file_path):
	# Given a file path , return feature matrix and target labels 
	phi = get_phi_basis_1(file_path)
	with open(file_path, 'r') as f:
		reader = csv.reader(f)
		next(reader)
		x = list(reader)
	row_length = int(len(x))
	y = []
	for row in x:
		#datetime = sr.dt.strptime('% B % d, % Y, % r')
		fare = float(row[6])/170
		y.append(fare)	
		#np.append(([1, hour, src_lat, src_long, dest_lat, dest_long, n_pass]))
	#print(phi)
	#print(y)
	y = np.array(y)
	return phi, y

def get_phi_basis_2(file_path):
	with open(file_path, 'r') as f:
		reader = csv.reader(f)
		next(reader)
		x = list(reader)
	row_length = int(len(x))
	phi = []
	for row in x:
		#datetime = sr.dt.strptime('% B % d, % Y, % r')
		dtime = row[0].split(':')
		dtime1 = dtime[0].split('-')
		year = dtime1[0]
		month = dtime1[1]
		day = dtime1[2].split()
		hour = int(day[1])
		day = day[0]
		## NORMALISATION
		if (hour > 12): ##changed here
			hour = np.exp((float(hour) - 12)/24.0)
		else:
			hour = np.exp((float(hour))/24.0)
		year = ((float(year)-2009)/6.0)
		#print(hour)
		#latitude 102 - 109, long 62 - 72, pass <=6
		src_lat = (float(row[1]) + 109)/7.0
		src_long = (float(row[2]) -62)/10.0
		dest_lat = (float(row[3]) + 109)/7.0
		dest_long = (float(row[4]) -62)/10.0
		n_pass = int(row[5]) 
		if (n_pass > 3):
			n_pass = 1
		else:
			n_pass= 0   #changed here 
		#dist = ((dest_lat-src_lat)**2 + (dest_long-src_long)**2)**0.5
		dist = np.absolute((dest_lat-src_lat)) + np.absolute((dest_long-src_long)); ##changed here
		phi.append([1, year, hour, dist, n_pass])
		#np.append(([1, hour, src_lat, src_long, dest_lat, dest_long, n_pass]))
	#print(phi)
	#print(y)
	phi = np.array(phi)
	return phi

def get_features_basis2(file_path):
	# Given a file path , return feature matrix and target labels 
	phi = get_phi_basis_1(file_path)
	with open(file_path, 'r') as f:
		reader = csv.reader(f)
		next(reader)
		x = list(reader)
	row_length = int(len(x))
	y = []
	for row in x:
		fare = float(row[6])/170
		y.append(fare)	
	y = np.array(y)
	return phi, y

def compute_RMSE(phi, w , y) :
	# Root Mean Squared Error
	print((phi@w).shape,y.shape)
	diff = y.reshape(-1,1)-(phi@w).reshape(-1,1)
	error = np.linalg.norm(diff, 2)
	return error

def generate_output(phi_test, w):
	# writes a file (output.csv) containing target variables in required format for Kaggle Submission.
	y_test = (phi_test@w)*170
	id_data = np.arange(len(y_test))
	id_data.astype(int)
	data_out = np.vstack([id_data, y_test.T])
	np.savetxt('output.csv', (data_out.T), delimiter = ',', header = "Id,fare", comments = "", fmt = '%i,%f') 
	return 0

def closed_soln(phi, y):
				# Function returns the solution w for Xw=y.
	return np.linalg.pinv(phi).dot(y)

def gradient_calc(phi, w, y):
	del_L = phi.T@phi@w - phi.T@np.expand_dims(y,axis=-1)
	return del_L*2/len(y)

def gradient_descent(phi, y) :
	# Mean Squared Error
	w = np.random.random((feature,1))
	grads = np.ones((feature,1))
	while(np.linalg.norm(grads, 2) > epsilon):
		print(np.linalg.norm(grads))
		grads = gradient_calc(phi, w, y)
		w = w - neta*grads
	return w

def sgd(phi, y) :
	# Mean Squared Error
	i = np.random.randint(low=1, high=len(y), size=1)
	i = i[0]
	w = np.random.random((feature,1))
	grads = np.ones((feature,1))
	while(np.linalg.norm(grads, 2) > epsilon):
		x_vec = phi[i:i+1,:]
		grads = 2*(x_vec.T@x_vec@w - x_vec.T*y[i]) # 7 x 1
		w = w - neta*grads
	return w

def pnorm_grad_calc(phi, w, y, p):
	if (p == 2):
		del_L = lamda*w - phi.T@(np.expand_dims(y,axis=-1) - phi@w)
	else:
		del_L = lamda*(np.linalg.norm(w))*w - phi.T@(np.expand_dims(y,axis=-1) - phi@w)
	return del_L*2/len(y)
def pnorm(phi, y, p) :
	# Mean Squared Error
	w = np.random.random((feature,1))
	grads = np.ones((feature,1))
	while(np.linalg.norm(grads, 2) > epsilon):
		print(np.linalg.norm(grads))
		grads = pnorm_grad_calc(phi, w, y, p)
		w = w - neta*grads
	return w	

def get_features_var(phi, y, size_l):
	w=gradient_descent(phi[:size_l,:], y[:size_l])
	return w
	
def main():
	
	
	#QUESTION 1
	phi, y = get_features('dev.csv')
	w1 = gradient_descent(phi,y)
	print(w1)
	rms = compute_RMSE(phi, w1, y)
	print(rms)
	w2 = closed_soln(phi,y)
	print(w2)
	rms2 = compute_RMSE(phi, w2, y)
	print(rms2)
	print("abs RMS diff of grad - closed:", rms-rms2)
	w3 = sgd(phi,y)
	print(w3)
	rms3 = compute_RMSE(phi,w3,y)
	print(rms3)
	print("abs RMS diff of grad - sgd:", rms3-rms)
	phi_test = get_phi('test.csv')
	generate_output(phi_test,w1)	
	
	
	##Question 2
	phi, y = get_features('dev.csv')
	w_p_2 = pnorm(phi,y,2)
	w_p_4 = pnorm(phi,y,4)
	print("wp2: ",w_p_2)
	print("wp4: ",w_p_4)
	rms_p_2 = compute_RMSE(phi, w_p_2, y)
	print("RMS 2-norm:",rms_p_2)
	rms_p_4 = compute_RMSE(phi, w_p_4, y)
	print("RMS 4-norm:",rms_p_4)
	phi_test = get_phi('test.csv')
	generate_output(phi_test,w_p_2)

	
	##QUEStion 3a
	phi, y = get_features_basis1('dev.csv')
	w_b1 = pnorm(phi,y, 2)
	print("w_b1: ", w_b1)
	rms_b1 = compute_RMSE(phi, w_b1, y)
	print("RMS w_b1: ", rms_b1)
	phi_test = get_phi_basis_1('test.csv')
	generate_output(phi_test,w_b1)
	
	
	
	##question 3b
	phi, y = get_features_basis2('dev.csv')
	w_b2 = pnorm(phi,y,2)
	print("w_b2: ", w_b2)
	rms_b2 = compute_RMSE(phi, w_b2, y)
	print("RMS w_b2: ", rms_b2)
	phi_test = get_phi_basis_2('test.csv')
	generate_output(phi_test,w_b2)
	
	
	
	#question 4
	phi,y = get_features_basis2('train.csv')
	limit = [10000, 15000, 20000, 30000, len(y)]
	rms_val = np.empty((5,1))
	for i in range(5):
		w=get_features_var(phi, y, limit[i])
		rms_val[i] = compute_RMSE(phi[:limit[i],:], w, y[:limit[i]])
	print(rms_val)	
	plt.plot(limit, rms_val)
	plt.xlabel('Subset sizes')
	plt.ylabel('RMS error')
	plt.title('RMS vs subset sizes')
	plt.show()
	
	
main()

