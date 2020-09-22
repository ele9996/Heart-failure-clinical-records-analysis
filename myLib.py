
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

def LL_Original(x_trainstd_fs,y_train_noPca,x_valstd_fs,y_val_noPca,x_teststd_fs,y_test_noPca):
  clf = LogisticRegression(random_state=0).fit(x_trainstd_fs,y_train_noPca.ravel())
  y_p_LR=clf.predict(x_valstd_fs)
  accuracy= accuracy_score(y_val_noPca,y_p_LR)
  #get the mean of the accuracies for each fold
  print("total accuracy of the model:",accuracy)      

  y_onTest_LR=clf.predict(x_teststd_fs)

  acc_on_Test= accuracy_score(y_test_noPca,y_onTest_LR)
  precision_on_Test = precision_score(y_test_noPca, y_onTest_LR, average='macro')
  recall_on_Test=recall_score(y_test_noPca, y_onTest_LR, average='macro')
  f1_on_Test=f1_score(y_test_noPca, y_onTest_LR, average='macro')
  print("Accuracy on test set={}".format(acc_on_Test))
  print("Precision on test set={}".format(precision_on_Test))
  print("Recall on test set={}".format(recall_on_Test)) 
  print("F1 score on test set={}".format(f1_on_Test))
  acc_original[2]=acc_on_Test
  f1_original[2]=f1_on_Test


  plot_Conf_Matrix(clf,x_teststd_fs,y_test_noPca.ravel())

def LL_smoteOnly(y_train_noPca,x_trainstd_fs,x_teststd_fs,y_test_noPca,x_valstd_fs,y_val_noPca):
  #apply smote
  unique, counts = np.unique(y_train_noPca, return_counts=True)
  minority_shape = dict(zip(unique, counts))[1]
  x1 = np.ones((minority_shape,x_trainstd_fs.shape[1]))
  k=0
  for i in range(0,x_trainstd_fs.shape[0]):
      
      if y_train_noPca[i] == 1:
          
          x1[k] =x_trainstd_fs[i]
          k = k + 1
  sampled_instances = SMOTE_100(x1)
  X_f = np.concatenate((x_trainstd_fs,sampled_instances), axis = 0)
  y_sampled_instances = np.ones(minority_shape)
  y_f = np.concatenate((y_train_noPca.ravel(),y_sampled_instances), axis=0)




  clf = LogisticRegression(random_state=0).fit(X_f,y_f)
  y_p_LR=clf.predict(x_valstd_fs)
  acc= accuracy_score(y_val_noPca,y_p_LR)

  #get the mean of the accuracies for each fold
  print("total accuracy of the model is:",acc)      

  y_onTest_LR=clf.predict(x_teststd_fs)

  acc_on_Test= accuracy_score(y_test_noPca,y_onTest_LR)
  precision_on_Test = precision_score(y_test_noPca, y_onTest_LR, average='macro')
  recall_on_Test=recall_score(y_test_noPca, y_onTest_LR, average='macro')
  f1_on_Test=f1_score(y_test_noPca, y_onTest_LR, average='macro')
  print("Accuracy on test set with smote={}".format(acc_on_Test))
  print("Precision on test set= with smote{}".format(precision_on_Test))
  print("Recall on test set with smote={}".format(recall_on_Test)) 
  print("F1 score on test set with smote={}".format(f1_on_Test))
  acc_smote[2]=acc_on_Test
  f1_smote[2]=f1_on_Test

  plot_Conf_Matrix(clf,x_teststd_fs,y_test_noPca.ravel())


def LL_KfoldOnly(kf10,cross_train_dataset,x_test_classifier_cross,y_test_noPca):
  fold = 0
  acc_array_LR=[0,0,0,0,0,0,0,0,0,0]
  for train_index, test_index in kf10.split(cross_train_dataset):
        X_train = cross_train_dataset.iloc[train_index].loc[:,selected_feat]
        X_val = cross_train_dataset.iloc[test_index][selected_feat]
        y_train = cross_train_dataset.iloc[train_index].loc[:,'DEATH_EVENT']
        y_val = cross_train_dataset.loc[test_index]['DEATH_EVENT']


        clf = LogisticRegression(random_state=0).fit(X_train,y_train)
        y_p_LR=clf.predict(X_val)
        acc_array_LR[fold]= accuracy_score(y_val,y_p_LR)
        fold=fold+1
  #get the mean of the accuracies for each fold
  print("total accuracy of the model after Cross Validation is:",mean(acc_array_LR) * 100)      

  y_onTest_LR=clf.predict(x_test_classifier_cross)

  acc_on_Test= accuracy_score(y_test_noPca,y_onTest_LR)
  precision_on_Test = precision_score(y_test_noPca, y_onTest_LR, average='macro')
  recall_on_Test=recall_score(y_test_noPca, y_onTest_LR, average='macro')
  f1_on_Test=f1_score(y_test_noPca, y_onTest_LR, average='macro')
  print("Accuracy on test set={}".format(acc_on_Test))
  print("Precision on test set={}".format(precision_on_Test))
  print("Recall on test set={}".format(recall_on_Test)) 
  print("F1 score on test set={}".format(f1_on_Test))
  acc_original_cv[2]=acc_on_Test
  f1_original_cv[2]=f1_on_Test

  plot_Conf_Matrix(clf,x_test_classifier_cross,y_test_noPca.ravel())

def LL_KfoldSmote(kf10,cross_train_dataset,x_test_classifier_cross,y_test_noPca):

  fold = 0
  acc_array_LR=[0,0,0,0,0,0,0,0,0,0]
  for train_index, test_index in kf10.split(cross_train_dataset):
        X_train = cross_train_dataset.iloc[train_index].loc[:,selected_feat].values
        X_val = cross_train_dataset.iloc[test_index][selected_feat].values
        y_train = cross_train_dataset.iloc[train_index].loc[:,'DEATH_EVENT'].values
        y_val = cross_train_dataset.loc[test_index]['DEATH_EVENT'].values

        #apply smote
        unique, counts = np.unique(y_train, return_counts=True)
        minority_shape = dict(zip(unique, counts))[1]
        x1 = np.ones((minority_shape, X_train.shape[1]))
        k=0
        for i in range(0,X_train.shape[0]):
            
            if y_train[i] == 1:
                
                x1[k] = X_train[i]
                k = k + 1
        sampled_instances = SMOTE_100(x1)
        X_f = np.concatenate((X_train,sampled_instances), axis = 0)
        y_sampled_instances = np.ones(minority_shape)
        y_f = np.concatenate((y_train,y_sampled_instances), axis=0)




        clf = LogisticRegression(random_state=0).fit(X_f,y_f)
        y_p_LR=clf.predict(X_val)
        acc_array_LR[fold]= accuracy_score(y_val,y_p_LR)
        fold=fold+1
  #get the mean of the accuracies for each fold
  print("total accuracy of the model after Cross Validation is:",mean(acc_array_LR) * 100)      

  y_onTest_LR=clf.predict(x_test_classifier_cross)

  acc_on_Test= accuracy_score(y_test_noPca,y_onTest_LR)
  precision_on_Test = precision_score(y_test_noPca, y_onTest_LR, average='macro')
  recall_on_Test=recall_score(y_test_noPca, y_onTest_LR, average='macro')
  f1_on_Test=f1_score(y_test_noPca, y_onTest_LR, average='macro')
  print("Accuracy on test set with smote={}".format(acc_on_Test))
  print("Precision on test set= with smote{}".format(precision_on_Test))
  print("Recall on test set with smote={}".format(recall_on_Test)) 
  print("F1 score on test set with smote={}".format(f1_on_Test))
  acc_smote_cv[2]=acc_on_Test
  f1_smote_cv[2]=f1_on_Test

  plot_Conf_Matrix(clf,x_test_classifier_cross,y_test_noPca.ravel())

