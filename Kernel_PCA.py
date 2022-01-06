import numpy as np
import os
from sklearn.decomposition import KernelPCA ### This does not work...
from sklearn.decomposition import PCA

def clean_data(file_name):
    data_trial_1 = []
    files = os.listdir(file_name)
    for file in files:
        trial1 = np.genfromtxt(file_name+"/{}".format(file),skip_header = 10,dtype=np.compat.unicode,delimiter = ",").transpose()
        c4_c3 = trial1[1]
        cz_fz = trial1[2]
        c4_c3[c4_c3 == ''] = 0.0
        cz_fz[cz_fz == ''] = 0.0
        c4_c3 = c4_c3.astype(float)
        cz_fz = cz_fz.astype(float)
        data_trial_1.append(np.asarray([c4_c3,cz_fz]))
    data_trial_1 = np.asarray(data_trial_1)
    return data_trial_1

def kernel_transformation(matrix,kernel):
    transformer = KernelPCA(n_components=1, kernel=kernel,fit_inverse_transform=True)
    X_transformed = transformer.fit_transform(matrix)
    return np.transpose(transformer.inverse_transform(X_transformed))
def PCA_(matrix):
    transformer = PCA(n_components=1)
    X_transformed = transformer.fit_transform(np.transpose(matrix))
    return np.transpose(transformer.inverse_transform(X_transformed))

def walsch_convolution(averaged_signal):
    return np.convolve([1,-1,-1,1],averaged_signal)

def clip(vector,min_,max_):
    temp = []
    for i in range(0,len(vector)):
        if i<min_ or i>max_:
            temp.append(0)
        else:
            temp.append(vector[i])
    return temp
def percent_time_difference(latency_1,latency_2):
    average = (latency_1+latency_2)/2
    difference = np.abs(latency_1-latency_2)
    return 100* difference/average
def peak_to_peak_percentage(peak_difference_1,peak_difference_2):
    average = (peak_difference_1+peak_difference_2)/2
    difference = np.abs(peak_difference_1-peak_difference_2)
    return 100*difference/average

def K(Matrix,Sigma):
    K = np.ones((len(np.transpose(Matrix)),len(np.transpose(Matrix))))
    for x_i in Matrix:
        K+= np.outer(phi(x_i,Sigma),phi(x_i,Sigma))
    return K/len(Matrix)
def k_ij(X,Y,Sigma):
    x = np.exp(-np.linalg.norm(X-Y)**2/(2*Sigma**2))
    x = np.abs(x)
    return x

def Gram_Matrix(K):
    N_1 = np.matrix(1/len(K) * np.ones((len(K),len(K))))
    K = np.matrix(K)
    K_Bar = K-N_1*K-K*N_1+N_1*K*N_1
    return K_Bar
def keep_n_components(eigenvalues,eigenvectors,components_to_keep=1):
    
    #largest_to_smallest = np.argsort(eigenvalues)[::-1]
    #eigenvalues = np.delete(eigenvalues,largest_to_smallest[components_to_keep:len(eigenvalues)])
    #eigenvectors = np.delete(eigenvectors,largest_to_smallest[components_to_keep:len(eigenvectors)],axis=0)
    largest = np.argmax(eigenvalues)
    return np.array([eigenvalues[largest]]),np.array(eigenvectors[largest])

def inverse_function(initial_z,max_iterations,eigenvalues,eigenvectors,original_matrix,sigma,components_to_keep=1):
    z=initial_z
    Trials = len(original_matrix)
    for k in range(0,max_iterations):
        init_z = z
        principal_components = []
        for eigs in eigenvectors:
            principal_components.append(k_ij(z,eigs,sigma))
        numerator_sum = np.zeros(len(np.transpose(original_matrix)))
        for i in range(0,Trials):
            gamma = gamma_i(principal_components,eigenvectors,i,components_to_keep)
            k = k_ij(z,original_matrix[i],sigma)
            x = original_matrix[i]
            k = np.abs(k)
            numerator_sum += gamma*k*x
        denominator = 0
        for j in range(0,Trials):
            k = k_ij(z,original_matrix[i],sigma)
            k = np.abs(k)
            gamma = gamma_i(principal_components,eigenvectors,i,components_to_keep)
            denominator += k*gamma
        if denominator == 0:
            return z
        z = numerator_sum/denominator
    return z

def y_k(X,input_,Sigma):
    return np.dot(phi(input_[0],Sigma),phi(X,Sigma))
    
def phi(X,Sigma):
    return np.exp(-np.power(X,2)/(2*Sigma**2))

def gamma_i(principal_components,eigenvectors,i,components_to_keep=1):
    sum = 0
    principal_components,eigenvectors = keep_n_components(principal_components,eigenvectors,components_to_keep=1)
    for k in range(0,len(principal_components)):
        eig_vec = np.array(eigenvectors)[k][i]
        eigvals = principal_components[k]
        sum+=eigvals*eig_vec
    return np.abs(sum)

