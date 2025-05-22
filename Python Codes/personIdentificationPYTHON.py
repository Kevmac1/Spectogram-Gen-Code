import numpy as np
from scipy.special import rel_entr

#freq distribution training data 
Kathryn1_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kathryn1.npy")
Kathryn2_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kathryn2.npy")
Kathryn4_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kathryn4.npy")
Kathryn5_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kathryn5.npy")

Fei1_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Fei1.npy")
Fei2_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Fei2.npy")
Fei4_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Fei4.npy")
#Fei5_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Fei5.npy")

Alex1_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Alex1.npy")
Alex2_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Alex2.npy")
Alex4_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Alex4.npy")
Alex5_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Alex5.npy")

#Kevin1_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kevin1.npy")
Kevin2_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kevin2.npy")
#Kevin4_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kevin4.npy")
Kevin5_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kevin5.npy")

Leon1_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Leon1.npy")
Leon2_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Leon2.npy")
Leon4_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Leon4.npy")
Leon5_P = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Leon5.npy")


#find avg freq dist for each person
KathrynStack = np.stack([Kathryn1_P, Kathryn2_P, Kathryn4_P, Kathryn5_P])
Kathryn_P = np.mean(KathrynStack, axis=0)

FeiStack = np.stack([Fei1_P, Fei2_P, Fei4_P]) #Fei5
Fei_P = np.mean(FeiStack, axis=0)

AlexStack = np.stack([Alex1_P, Alex2_P, Alex4_P, Alex5_P])
Alex_P = np.mean(AlexStack, axis=0)

KevinStack = np.stack([Kevin2_P, Kevin5_P]) #Kevin1 #Kevin4
Kevin_P = np.mean(KevinStack, axis=0)

LeonStack = np.stack([Leon1_P, Leon2_P, Leon4_P, Leon5_P])
Leon_P = np.mean(LeonStack, axis=0)

#freq distribution testing data
#Kathryn3_Q = np.array([0.0195, 0.0252, 0.0293, 0.0200, 0.0156, 0.0148, 0.0138, 0.0122, 0.0113, 0.0113, 0.0120, 0.0120, 0.0126, 0.0121, 0.0121, 0.0131, 0.0133, 0.0142, 0.0149, 0.0147, 0.0138, 0.0130, 0.0126, 0.0118, 0.0112, 0.0108, 0.0110, 0.0116])
Kathryn3_Q = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kathryn3.npy")
Fei3_Q = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Fei3.npy")
Alex3_Q = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Alex3.npy")
Kevin3_Q = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Kevin4.npy") #Kevin3
Leon3_Q = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/freq_dist/freq_dist_vector_Leon3.npy")

#KLD

Q = Alex3_Q #change this to identify different people 


kld1 = np.sum(rel_entr(Kathryn_P, Q))
kld2 = np.sum(rel_entr(Fei_P, Q))
kld3 = np.sum(rel_entr(Alex_P, Q))
kld4 = np.sum(rel_entr(Kevin_P, Q))
kld5 = np.sum(rel_entr(Leon_P,Q))

klds = np.array([kld1, kld2, kld3, kld4, kld5])
# Normalize KLDs 
klds = klds * (3/np.max(klds))
# Assign back to original variable names
kld1, kld2, kld3, kld4, kld5 = klds[0], klds[1], klds[2], klds[3], klds[4]

print("KLD Kathryn:", kld1)
print("KLD Fei:", kld2)
print("KLD Alex:", kld3)
print("KLD Kevin:", kld4)
print("KLD Leon:", kld5)



#training data 4 vectors (8D)
Kathryn1 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kathryn1.npy")
Kathryn2 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kathryn2.npy")
Kathryn4 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kathryn4.npy")
Kathryn5 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kathryn5.npy")

Fei1 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Fei1.npy")
Fei2 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Fei2.npy")
Fei4 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Fei4.npy")
#Fei5 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Fei5.npy")

Alex1 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Alex1.npy")
Alex2 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Alex2.npy")
Alex4 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Alex4.npy")
Alex5 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Alex5.npy")

#Kevin1 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kevin1.npy")
Kevin2 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kevin2.npy")
#Kevin4 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kevin4.npy")
Kevin5 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kevin5.npy")

Leon1 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Leon1.npy")
Leon2 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Leon2.npy")
Leon4 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Leon4.npy")
Leon5 = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Leon5.npy")


# Combine all vectors into a single matrix
#Adjust when we get more data
data = np.array([
    Kathryn1, Kathryn2, Kathryn4, Kathryn5,
    Fei1, Fei2, Fei4,
    Alex1, Alex2, Alex4, Alex5,
    Kevin2, Kevin5,
    Leon1, Leon2, Leon4, Leon5
])

# Normalize (z-score)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normalized_data = (data - mean) / std

# Assign back to original variable names
Kathryn1, Kathryn2, Kathryn4, Kathryn5 = normalized_data[0:4]
Fei1, Fei2, Fei4 = normalized_data[4:7] #Fei5
Alex1, Alex2, Alex4, Alex5 = normalized_data[7:11]
Kevin2, Kevin5 = normalized_data[11:13]
Leon1, Leon2, Leon4, Leon5 = normalized_data[13:17]

#compute centroid
points_1 = np.stack([Kathryn1, Kathryn2, Kathryn4, Kathryn5])
centroid_1 = np.mean(points_1, axis=0)

points_2 = np.stack([Fei1, Fei2, Fei4]) #Fei5
centroid_2 = np.mean(points_2, axis=0)

points_3 = np.stack([Alex1, Alex2, Alex4, Alex5])
centroid_3 = np.mean(points_3, axis=0)

points_4 = np.stack([Kevin2, Kevin5]) #Kevin1 #Kevin4
centroid_4 = np.mean(points_4, axis=0)

points_5 = np.stack([Leon1, Leon2, Leon4, Leon5])
centroid_5 = np.mean(points_5, axis=0)


#test data
#test_Kathryn = np.array([27.9462, 1.2552, 0.4533, 0.5690, -0.0006, 0.0255, -0.0041, 0.2246])
test_Kathryn = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kathryn3.npy")
test_Fei = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Fei3.npy")
test_Alex = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Alex3.npy")
test_Kevin = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Kevin4.npy") #Kevin3
test_Leon = np.load("/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/features/feature_vector_Leon3.npy")


data_test = np.array([
    test_Kathryn, test_Fei, test_Alex, test_Kevin, test_Leon
])

# Normalize (z-score)
mean_test = np.mean(data_test, axis=0)
std_test = np.std(data_test, axis=0)
normalized_data_test = (data_test - mean_test) / std_test

# Assign back to original variable names
test_Kathryn, test_Fei, test_Alex, test_Kevin, test_Leon = normalized_data_test[0:5]




test = test_Alex


#compute Euclidean distance
#dist from text centroid to fixed centroids
distance_1 = abs(np.linalg.norm(np.append(test - centroid_1,kld1))) #dist from Kathryn
distance_2 = abs(np.linalg.norm(np.append(test - centroid_2,kld2))) #dist from Fei
distance_3 = abs(np.linalg.norm(np.append(test - centroid_3,kld3))) #dist from Alex
distance_4 = abs(np.linalg.norm(np.append(test - centroid_4,kld4))) #dist from Kevin
distance_5 = abs(np.linalg.norm(np.append(test - centroid_5,kld5))) #dist from Leon

#results
print("Distance from Kathryn:", distance_1)
print("Distance from Fei:", distance_2)
print("Distance from Alex:", distance_3)
print("Distance from Kevin:", distance_4)
print("Distance from Leon:", distance_5)
#print("Test Vector:", test)


closeness_bound = 2.1 #tweak
if distance_1 > closeness_bound and distance_2 > closeness_bound and distance_3 > closeness_bound and distance_4 > closeness_bound and distance_5 > closeness_bound:
    print("no matches to training data")
elif distance_1 < distance_2 and distance_1 < distance_3 and distance_1 < distance_4 and distance_1 < distance_5:
    print("more likely Kathryn")
elif distance_2 < distance_1 and distance_2 < distance_3 and distance_2 < distance_4 and distance_2 < distance_5:
    print("more likely Fei")
elif distance_3 < distance_1 and distance_3 < distance_2 and distance_3 < distance_4 and distance_3 < distance_5:
    print("more likely Alex")
elif distance_4 < distance_1 and distance_4 < distance_2 and distance_4 < distance_3 and distance_4 < distance_5:
    print("more likely Kevin")
else:
    print("more likely Leon")