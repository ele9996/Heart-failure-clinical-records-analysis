
def nearest_neighbour(X, x):
    euclidean = np.ones(X.shape[0]-1)
    
    additive = [None]*(1*X.shape[1])
    additive = np.array(additive).reshape(1, X.shape[1])
    k = 0
    for j in range(0,X.shape[0]):
        if np.array_equal(X[j], x) == False:
            euclidean[k] = sqrt(sum((X[j]-x)**2))
            k = k + 1
    euclidean = np.sort(euclidean)
    weight = rand.random()
    while(weight == 0):
        weight = rand.random()
    additive = np.multiply(euclidean[:1],weight)
    return additive
    
def SMOTE_100(X):
    new = [None]*(X.shape[0]*X.shape[1])
    new = np.array(new).reshape(X.shape[0],X.shape[1])
    k = 0
    for i in range(0,X.shape[0]):
        additive = nearest_neighbour(X, X[i])
        for j in range(0,1):
            new[k] = X[i] + additive[j]
            k = k + 1
    return new 
  
def plot_selFeat(selectFeat, allFeat ):
  #1 --> true
  #0 --> false


  d = {} 
  
  # iterating through the elements of list 
  for i in allFeat: 
    d[i] = None
    
  
  for feat, val in d.items():
    d[feat]=0
    for f in selectFeat:
      if (feat==f):
        
        d[feat]=1
        break
        
    
    

  

  data_items = d.items()
  data_list = list(data_items)

  

  col= ["features","taken"]

  seldf = pd.DataFrame(data=data_list, columns=col)
  
  
  plot=sns.catplot(x="taken",y="features", kind="bar", data =seldf)

  return plot

  

 
