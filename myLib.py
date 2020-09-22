
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

  #Useful function for k-fold definition
def pr_N_mostFrequentfeat(arr, n, k): 
    sel_feat=[]
    um = {} 
    for i in range(n): 
        if arr[i] in um: 
            um[arr[i]] += 1
        else: 
            um[arr[i]] = 1
    a = [0] * (len(um)) 
    j = 0
    for i in um: 
        a[j] = [i, um[i]] 
        j += 1
    a = sorted(a, key = lambda x : x[0], 
                         reverse = True) 
    a = sorted(a, key = lambda x : x[1],  
                         reverse = True) 
                           
    # display the top k numbers  
    
    for i in range(k): 
        #print(a[i][0], end = " ")
        sel_feat.append(a[i][0])
    
    
    return sel_feat

def kfoldize2(kf, rn, shift=.1):
    train = pd.DataFrame()
    test = pd.DataFrame()
    i = 1
    for train_index, test_index in kf.split(rn):
        train_df = pd.DataFrame(np.take(rn, train_index), columns=["x"])
        train_df["val"] = i - shift
        train = train.append(train_df)

        test_df = pd.DataFrame(np.take(rn, test_index), columns=["x"])
        test_df["val"] = i + shift
        test = test.append(test_df)
        i += 1
    return train, test

def plot_Conf_Matrix(classifier,X_te,y_te):
 
  title= "Confusion matrix"
  class_names= ["Alive","Dead"]
  disp = plot_confusion_matrix(classifier, X_te, y_te,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize="true")
  disp.ax_.set_title(title)

  print(title)
  plt.show()

 
