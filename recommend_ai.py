# code
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import json
from urllib.request import urlopen
  
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

  
def filterData(views, products):
    n_views = len(views)
    n_products = len(views['product_id'].unique())
    n_customers = len(views['customer_id'].unique())
    
    print(f"Number of views: {n_views}")
    print(f"Number of unique productId's: {n_products}")
    print(f"Number of unique customers: {n_customers}")
    print(f"Average views per customer: {round(n_views/n_customers, 2)}")
    print(f"Average views per product: {round(n_views/n_products, 2)}")
    
    user_freq = views[['customer_id', 'product_id']].groupby('customer_id').count().reset_index()
    user_freq.columns = ['customer_id', 'n_views']
    user_freq.head()
    
    
    # Find Lowest and Highest rated movies:
    mean_view = views.groupby('product_id')[['total']].mean()
    # Lowest rated movies
    lowest_viewed = mean_view['total'].idxmin()
    products.loc[products['id'] == lowest_viewed]
    # Highest rated movies
    highest_viewed = mean_view['total'].idxmax()
    products.loc[products['id'] == highest_viewed]
    # show number of people who rated movies rated movie highest
    views[views['product_id']==highest_viewed]
    # show number of people who rated movies rated movie lowest
    views[views['product_id']==lowest_viewed]
    
    ## the above movies has very low dataset. We will use bayesian average
    product_stats = views.groupby('product_id')[['total']].agg(['count', 'mean'])
    product_stats.columns = product_stats.columns.droplevel()
    return views, products
  
#Now, we create user-item matrix using scipy csr matrix
from scipy.sparse import csr_matrix
  
def create_matrix(df):
      
    N = len(df['customer_id'].unique())
    M = len(df['product_id'].unique())
      
    # Map Ids to indices
    customer_mapper = dict(zip(np.unique(df["customer_id"]), list(range(N))))
    product_mapper = dict(zip(np.unique(df["product_id"]), list(range(M))))
      
    # Map indices to IDs
    customer_inv_mapper = dict(zip(list(range(N)), np.unique(df["customer_id"])))
    product_inv_mapper = dict(zip(list(range(M)), np.unique(df["product_id"])))
      
    customer_index = [customer_mapper[i] for i in df['customer_id']]
    product_index = [product_mapper[i] for i in df['product_id']]
  
    X = csr_matrix((df["total"], (product_index, customer_index)), shape=(M, N))
      
    return X, customer_mapper, product_mapper, customer_inv_mapper, product_inv_mapper
  

  
from sklearn.neighbors import NearestNeighbors
"""
Find similar products using KNN
"""
def find_similar_products(product_mapper,product_inv_mapper,product_id, X, k, metric='cosine', show_distance=False):
      
    neighbour_ids = []
      
    product_ind = product_mapper[product_id]
    product_vec = X[product_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    product_vec = product_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(product_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(product_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids
  
  



def recommend(productId):
    # Get dataframe of list product
    url = "https://tech-api.herokuapp.com/v1/product/client-list"
    response = urlopen(url)
    data_product_json = json.loads(response.read())
    data_product_file = data_product_json['data']['data']
    products = pd.json_normalize(data_product_file, max_level=1)

    # Get dataframe of list customer view product
    url2 = "https://tech-api.herokuapp.com/v1/product/customer-view"
    response2 = urlopen(url2)
    data_json = json.loads(response2.read())
    data_file = data_json['data']['data']
    views = pd.json_normalize(data_file, max_level=1)
    views, products = filterData(views,products)
    X, customer_mapper, product_mapper, customer_inv_mapper, product_inv_mapper = create_matrix(views)
    similar_ids = find_similar_products(product_mapper,product_inv_mapper,productId, X, k=len(views) - 1)
    data = ""
    for i in similar_ids:
        data = data + str(i) + ","
    return data

