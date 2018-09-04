"""
@author: marpavakav 
This script is based on the SMOTE oversampling technique as Chawla et al. explain with some additional functionality
https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/chawla2002.html

Additional functionality: For smote_ratio>nn we added the functionality to use all nn neighbours and repeat 
the process by: times=int(smote_ratio/nn)

Inputs:
1. input_data= the data of the minority class (dataframe)
2. nn= number of nearest neighbours to use (integer)
3. smote_ratio=how much oversampling to do (integer in multiples of 1 
   e.g. smote_ratio=2 means smote data produced will be 2x times the size of input_data) 
4. plot_flag=produce plots of first 2 parameters of the original and smote data (1-yes, 0-no)

Descritpion:
The nn nearest neighbours are found for each input data point. 
A number of them is randomly selected (this number is equal to the smote_ratio) for each input data point.
The new smote data point will be in a random position between each input data point and each selected neighbour
Note: For smote_ratio>nn we added the functionality to use all nn neighbours and repeat the process by: 
times=int(smote_ratio/nn) 

"""
def mysmote(input_data,nn,smote_ratio, plotflag):
    #import libraries
    import numpy as np, pandas as pd, random, matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors

    #check if nn and smote_ratio are integers 
    if isinstance(nn, int)==False:
        raise TypeError('Variable nn should be integer')
    if isinstance(smote_ratio, int)==False:
        raise TypeError('Variable smote_ratio should be integer')
    #check if input_data is a dataframe
    if isinstance(input_data, pd.DataFrame)==False:
        raise TypeError('Variable input_data should be dataframe')
    #if smote_ratio<nn then iterate only once else iterate enough times to create smote_ratio data
    if smote_ratio<=nn:
        totalloops=1
        n_nb=smote_ratio
    else:
        totalloops=int(smote_ratio/nn)
        n_nb=nn
    
    #identify the nearest neighbours of each point
    #nn+1 instead of nn because first column will be removed
    nbrs = NearestNeighbors(n_neighbors=nn+1).fit(input_data) 
    distances, indices = nbrs.kneighbors(input_data)
    #remove first column because the first nearest neighbour is the same point 
    distances=distances[:,1:nn+1]
    indices=indices[:,1:nn+1]
    #initialise smotedata dataframe
    smotedata = pd.DataFrame(columns=input_data.columns)
    #loop through all minority samples
    for dataline in range(0, len(input_data)):
        #shuffle the neighbours
        niter=np.arange(0,nn)
        np.random.shuffle(niter)
        #take the first n_nb neighbours
        niter=niter[0:n_nb]
        temp_neighbours=indices[dataline,niter]
    
        for irepeat in range(1,totalloops+1):                         
          for iteration in range(1,n_nb+1):
              #find distance of each d point
              tempdistance=input_data.iloc[temp_neighbours[iteration-1],:]-input_data.iloc[dataline,:]
              #find a random point between this distance
              rand=random.uniform(0, 1)
              tempdistance=rand*tempdistance
              newpoint=input_data.iloc[dataline,:]+tempdistance
              smotedata=smotedata.append(newpoint,ignore_index=True)

    if plotflag==1:
        #plot original data and smotedata only first 2 parameters
        plt.figure()
        plt.plot(smotedata.iloc[:,0], smotedata.iloc[:,1], 'ro',label='smote data')
        plt.plot(input_data.iloc[:,0], input_data.iloc[:,1], 'bo',label='original data')
        plt.xlabel('parameter 1')
        plt.ylabel('parameter 2')
        plt.legend(loc='best')
        plt.show()
      
    return smotedata