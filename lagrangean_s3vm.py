from __future__ import division
from sklearn.svm import SVC
import sklearn.utils as skutils
from scipy.sparse import vstack
from scipy.sparse.csr import csr_matrix
import numpy as np
from utils import *

def lagrangian_s3vm_train(l,
                          u,
                          xtrain_l,
                          ytrain_l,
                          xtrain_u,
                          C=1.0,
                          gamma=1/2,
                          kernel='rbf',
                          r=0.5,
                          balance_tolerance=.0,
                          major_iters=10,
                          minor_iters=100,
                          lam_0=0.0,
                          theta_0=1.0,
                          batch_size=2000,
                          C_star_sequence = [0.1, 0.25, 0.5, 1.0],
                          rdm=np.random.RandomState()):
    #initial fitting
    svc = SVC(C=C, gamma=gamma, kernel=kernel)
    svc.fit(xtrain_l, ytrain_l)
    #initialization
    max_violation = u*balance_tolerance
    beta = u*(2*r - 1)#constant term
    C_star_sequence = C_star_sequence
    for t in xrange(major_iters):
        #############################################
        #compute unknown labels
        distances = svc.decision_function(xtrain_u)
        y_u = lagrangian_heuristic(distances,
                                    max_violation,
                                    beta,
                                    minor_iters,
                                    lam_0,
                                    theta_0)            
        #refit
        xtrain_u_shuffled, ytrain_u_shuffled = skutils.shuffle(xtrain_u, y_u)
        if type(xtrain_l) == csr_matrix:
            xtrain = vstack([xtrain_l, xtrain_u_shuffled[:batch_size - l]])
        else:
            xtrain = np.concatenate((xtrain_l, xtrain_u_shuffled[:batch_size - l])) 
        ytrain = np.concatenate((ytrain_l, ytrain_u_shuffled[:batch_size - l]))
        ############################################# 
        sample_weight = l*[1]+u*[C_star_sequence[t]]
        sample_weight = sample_weight[:batch_size]
        svc.fit(xtrain,
                ytrain,
                sample_weight=sample_weight)            
        #check termination condition (the last refit is always computed with the last C* sequence)         
        if t == len(C_star_sequence) - 1 : break            
    return svc

#returns the best labeling of the unlabeled points wrt to
# - their distance from the separating hyperplane
# - the relaxed balance constraint
def lagrangian_heuristic(distances,
                         max_violation,
                         beta,
                         iterations,
                         lam_0,
                         theta_0):
    lam = lam_0
    theta = theta_0
    y_a, y_b = None, None
    for k in xrange(iterations):
        #get the best labeling with current lambda
        best_labels = map(lambda dist : get_best_label(dist, lam), distances)
        #collect the constraint violation
        violation = sum(best_labels) - beta        
        #termination condition check
        if abs(violation) <= max_violation or k==iterations-1:  
            break
        if violation < 0 : y_a = best_labels  
        if violation > 0 : y_b = best_labels
        if y_a is not None and y_b is not None:
            #update lambda with planes intersection
            lam = planes_intersection(y_a, y_b, distances, beta)
        else:
            #update lambda wrt the violation sign
            lam += theta*violation
            #decrement theta according to the decay rule
            theta = update_theta(theta) 
    return best_labels
    
def planes_intersection(y_a, y_b, distances, beta):
    u = len(y_a)
    emp_err_on_y_a = sum([hinge(distances[i]*y_a[i]) for i in xrange(u)])
    emp_err_on_y_b = sum([hinge(distances[i]*y_b[i]) for i in xrange(u)])
    numerator = emp_err_on_y_b - emp_err_on_y_a
    #
    violation_on_y_a = sum(y_a) - beta
    violation_on_y_b = sum(y_b) - beta
    denominator = violation_on_y_a - violation_on_y_b    
    lam_next = numerator/denominator  
    return lam_next
    
def get_best_label(distance, lam):
    plus_1 = hinge(distance) + lam
    minus_1 = hinge(-distance) - lam
    return 1.0 if plus_1 < minus_1 else -1.0

def update_theta(theta):
    return theta*0.9

def hinge(t):
    return max(0, 1-t)
