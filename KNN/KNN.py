###############################
### DO NOT CHANGE THIS CELL ###
###############################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
##import imageio
##from tqdm import tqdm_notebook as tqdm

print('Version information')

print('python: {}'.format(sys.version))
print('matplotlib: {}'.format(matplotlib.__version__))
print('numpy: {}'.format(np.__version__))

# Set random seed so output is all same
np.random.seed(1)


def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
        dist: N x M array, where dist2[i, j] is the euclidean distance between
        x[i, :] and y[j, :]
    """
    # raise NotImplementedErro
    return np.sqrt((np.square(x[:, np.newaxis] - y).sum(axis=2)))

if 0:
    # Check answer
    np.random.seed(1)
    x = np.random.randn(2, 2)
    y = np.random.randn(3, 2)

    print("*** Expected Answer ***")
    print("""==x==
    [[ 1.62434536 -0.61175641]
     [-0.52817175 -1.07296862]]
    ==y==
    [[ 0.86540763 -2.3015387 ]
     [ 1.74481176 -0.7612069 ]
     [ 0.3190391  -0.24937038]]
    ==dist==
    [[1.85239052 0.19195729 1.35467638]
     [1.85780729 2.29426447 1.18155842]]""")

    print("\n*** My Answer ***")
    print("==x==")
    print(x)
    print("==y==")
    print(y)
    print("==dist==")
    print(pairwise_dist(x, y))


def softmax(logits):
    # raise NotImplementedError
    e_logits = np.exp(logits - np.max(logits, axis=1)[:, np.newaxis])
    #print(logits - np.max(logits, axis=1)[:, np.newaxis])
    return e_logits / np.sum(e_logits, axis=1)[:, np.newaxis]

if 0:
    logits = np.array([[1000, 1000],
                       [1, 2]], dtype=np.float32)
    print("""Correct answer:
    ===softmax===
    [[0.5        0.5       ]
     [0.26894143 0.7310586 ]]""")

    print("My answer:")
    print(softmax(logits))


def logsumexp(logits):
    """
    Args:
        logits: N x D numpy array
    Return:
        s: N x 1 array where s[i,0] = logsumexp(logits[i,:])
    """
    ##raise NotImplementedError

    e_logits = np.exp(logits - np.max(logits, axis=1)[:, np.newaxis])
    return np.add(np.log(np.sum(e_logits, axis=1))[:, np.newaxis], np.max(logits, axis=1)[:, np.newaxis])

if 0:
    logits = np.array([[1000, 1000],
                       [1, 2]], dtype=np.float32)
    print("""Correct Answer:
    ===logsumexp===
    [[1000.6932   ]
     [   2.3132617]]""")

    print("My answer: ")
    print(logsumexp(logits))


# below are some helper functions for plot.
# you don't have to modify them.

def plot_images(img_list, title_list, figsize=(11, 6)):
    assert len(img_list) == len(title_list)
    fig, axes = plt.subplots(1, len(title_list), figsize=figsize)
    for i, ax in enumerate(axes):
        ax.imshow(img_list[i] / 255.0)
        ax.set_title(title_list[i])
        ax.axis('off')

    plt.show()


def plot_scatter(samples, ids):
    colors = np.zeros((len(ids), 3))
    choices = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    num_points = []
    for i in range(3):
        num_points.append(np.sum(ids == i))
    maps = np.argsort(num_points)
    for i in range(3):
        colors[np.where(ids == maps[i]), :] = choices[i]
    plt.scatter(samples[:, 0], samples[:, 1], s=1, color=colors)
    plt.axis('equal')


## kmeans
class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def _init_centers(self, points, K, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        """
        ##raise NotImplementedError

        D = points.shape[1]
        centers = np.zeros([K,D])

        for i in range(D):
            #print(np.min(points[:,i]), np.max(points[:,i]))
            centers[:,i] = np.random.uniform(low=np.min(points[:,i]), high=np.max(points[:,i]), size=[K, 1]).squeeze()
        #print(centers)
        return centers

    def _update_assignment(self, centers, points):
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        # raise NotImplementedError
        EuclidianDistance = pairwise_dist(centers, points)

        cluster_idx = np.argmin(EuclidianDistance, axis=0)
        return cluster_idx

    def _update_centers(self, old_centers, cluster_idx, points):
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """
        # raise NotImplementedError
        idx_array = np.zeros([old_centers.shape[0], points.shape[0]])
        idx_array[cluster_idx, np.arange(points.shape[0])] = 1
        # print(idx_array)
        points_each_cluster = np.sum(idx_array, axis=1)
        new_centers = np.dot(idx_array, points)
        for i in range(old_centers.shape[0]):
            if points_each_cluster[i]  > 0:
                new_centers[i,:] = new_centers[i,:]/points_each_cluster[i]
            else:
                new_centers[i, :] = old_centers[i,:]

        return new_centers

    def _get_loss(self, centers, cluster_idx, points):
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.

        Hint: The loss equals to the average squared distance for all the observation points.
              You could call pairwise_dist() function which you have implemented to get the distance.
        """
        # raise NotImplementedError
        dis = pairwise_dist(centers, points)
        idx_array = np.zeros([centers.shape[0], points.shape[0]])
        idx_array[cluster_idx, np.arange(points.shape[0])] = 1
        loss = np.sum(dis * idx_array) / (points.shape[0] * points.shape[1])
        return loss

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers

        Hint: You do not need to change it. For each iteration, we update the centers and calculate the loss.
        If the loss between two iterations qualify our two conditions, then we will stop the iteraion and return our centers.
        """
        centers = self._init_centers(points, K, **kwargs)
        #pbar = tqdm(range(max_iters))
        #for it in pbar:
        for it in range(100):
            cluster_idx = self._update_assignment(centers, points)

            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    print(it)
                    break
            prev_loss = loss
            #pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers

##GMM
class GMM(object):
    def __init__(self): # No need to implement
        pass
        
    def _init_components(self, points, K, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxD numpy array, the diagonal standard deviation of each gaussian.
            
        Hint: You could use the K-means results to initial GMM. It will help to converge. 
        For instance, you could use ids, mu = KMeans()(points, K)  to initialize.
        """
        #raise NotImplementedError
        km = KMeans()
        ids, mu = km(points, K)
        gamma = np.zeros([K, points.shape[0]]).T
        gamma[ np.arange(points.shape[0]),ids] = 1
        N_k = np.sum(gamma,axis=0)
        pi = N_k/np.sum(N_k)
        
        #for k in range(K):
        #    if N_k[k] >0:
        #        sigma[k] = sigma[k] + 1 / N_k[k]*np.sum(tao*np.sum((points - mu[:,np.newaxis,:])*(points - mu[:,np.newaxis,:]),axis=2)),axis=1)
        #sigma = 1 / N_k * np.sum(gamma.T*np.sum((points - mu[:,np.newaxis,:])*(points - mu[:,np.newaxis,:]),axis=2),axis=1)
        #var =
        N_k_temp = N_k
        N_k_temp[N_k==0] =1
        sigma = np.sqrt(1 / N_k_temp * np.sum(
            gamma * ((points - mu[:, np.newaxis, :]) * (points - mu[:, np.newaxis, :])).T, axis=1))
        sigma[sigma==0] = 1
        return pi, mu, sigma.T


    def _ll_joint(self, points, pi, mu, sigma):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
            
        Hint: Assume that the three dimensions of our multivariate gaussian are independent.  
              This allows you to write treat it as a product of univariate gaussians.
        """
        #raise NotImplementedError
        x_mean_2 = np.square(points - mu[:,np.newaxis,:])
        sigma_2 = -1/(2*sigma**2)
        sigma_1 = 1/np.sqrt(2*3.14*sigma)
        independent_conditional_prob = sigma_1*np.exp(sigma_2*np.transpose(x_mean_2,(1,0,2)))
        combined_conditional_prob = np.ones([independent_conditional_prob.shape[0],independent_conditional_prob.shape[1]])
        for i in range(independent_conditional_prob.shape[-1]):
            combined_conditional_prob = combined_conditional_prob*independent_conditional_prob[:,:,i]
        ll = np.log(pi+ 1e-150)+np.log(combined_conditional_prob+1e-150)
        #loss = np.dot(combined_conditional_prob, pi)
        
        return ll

    def _E_step(self, points, pi, mu, sigma):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        #raise NotImplementedError
        x_mean_2 = np.square(points - mu[:,np.newaxis,:])
        sigma_2 = -1/(2*sigma**2)
        sigma_1 = 1/(np.sqrt(2*3.14)*sigma)

        independent_conditional_prob = sigma_1*np.exp(sigma_2*np.transpose(x_mean_2,(1,0,2)))
        combined_conditional_prob = np.ones([independent_conditional_prob.shape[0],independent_conditional_prob.shape[1]])
        for i in range(independent_conditional_prob.shape[-1]):
            combined_conditional_prob = combined_conditional_prob*independent_conditional_prob[:,:,i]
        gamma =    (pi*combined_conditional_prob).T/np.sum(pi*combined_conditional_prob,axis=1)
        return gamma.T


    def _M_step(self, points, gamma):
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxD numpy array, the diagonal standard deviation of each gaussian. 
            
        Hint:  There are formulas in the slide.
        """
        #raise NotImplementedError
        N_k = np.sum(gamma,axis=0)
        N_k_temp = N_k
        N_k_temp[N_k==0] =1
        pi = N_k/points.shape[0]
        mu = (np.dot(gamma.T,points).T/N_k_temp).T

        sigma =np.sqrt( 1 / N_k_temp * np.sum(
            gamma * ((points - mu[:, np.newaxis, :]) * (points - mu[:, np.newaxis, :])).T, axis=1))
        sigma[sigma==0] = 1
        return pi,mu,sigma.T


    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxD numpy array), mu and sigma.
        
        Hint: You do not need to change it. For each iteration, we process E and M steps, then 
        """        
        pi, mu, sigma = self._init_components(points, K, **kwargs)
        
        pbar = tqdm(range(max_iters))
        for it in pbar:
        #for it in range(100):
            # E-step
            gamma = self._E_step(points, pi, mu, sigma)
            # M-step
            pi, mu, sigma = self._M_step(points, gamma)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(points, pi, mu, sigma)
            loss = -np.sum(logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            #print(loss)
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)


def knn(input,data,cluster_id,k):
    distance = pairwise_dist(input[np.newaxis,:],data)
    distance =distance.squeeze()

    id_array = []
    for i in range (k):
        min_index = np.argmin(distance)
        print(min_index)
        id_array.append(cluster_id[min_index])
        distance[min_index] = 1e10
    print(id_array)
if __name__ == "__main__":
    import csv
    data =[]
    with open('kc_house_data_processed_2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                data.append(row)
                line_count += 1
        print('Processed {line_count} lines.')
    data_array = np.asarray(data,dtype=float)




    ##data_array[:,1] =     data_array[:,1]/5
    ##data_array[:,4] =     data_array[:,4]/200
    ##data_array[:,5] =     data_array[:,5]/2000




    cluster_ids, centers = KMeans()(data_array[:,1:], K=5)
    with open('kc_house_data_processed_output_5.csv', mode='w') as employee_file:
            writer = csv.writer(employee_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(line_count-1):
                writer.writerow([data_array[i,0],cluster_ids[i]])

    """
    cluster_ids, centers = KMeans()(data_array[:,1:], K=16, max_iters=100)

    with open('kc_house_data_processed_output_16.csv', mode='w') as employee_file:
            writer = csv.writer(employee_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(line_count-1):
                writer.writerow([data_array[i,0],cluster_ids[i]])
    """

input =np.array( [3, 1,	1, 4,	8])
knn(input,data_array[:,1:],cluster_ids,5)