import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
import gc
import statistics
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
import csv

def customerSegmentation(filepath):
    # Part1: read in raw data with provided filepah
    aisle = pd.read_csv(filepath + '/aisles.csv')
    department = pd.read_csv(filepath + '/departments.csv')
    order_products = pd.read_csv(filepath + '/sample_order_products.csv')
    orders = pd.read_csv(filepath + '/sample_order.csv')
    products = pd.read_csv(filepath + '/products.csv')
    aisle_match = pd.read_csv(filepath + '/final_aisle_match.csv')
    
    # Part 2: data preprocessing
    
    # Take out orderID and productID from both train and prior table and merge them into one order_products_id table
    order_products_id = order_products[['order_id','product_id']]
    # take out orderID and userID from orders table
    order_user_id = orders[['order_id','user_id']]
    # get total order count for each user
    user_order_count = order_user_id[['user_id','order_id']]
    user_order_count = user_order_count.groupby(['user_id']).agg({'order_id': 'count'}).reset_index()
    user_order_count.rename(columns={"order_id": "order_count"}, inplace=True)
    #combine order_products and products table to see what aisle each product belongs for each order
    order_product_aisle = order_products_id.merge(products,on='product_id')
    #from order_product_aisle table, only select orderID and aisleID to aggregate some data since we only deal with 
    #aisle level granularity
    #In addition, count of products from each aisle for each order is kept
    order_aisle_id = order_product_aisle[['order_id','aisle_id']]
    order_aisle_id['aisle_id2'] = order_product_aisle['aisle_id']
    order_aisle_count = order_aisle_id.groupby(['order_id','aisle_id']).agg({'aisle_id2': 'count'}).reset_index()
    order_aisle_count.rename(columns={"aisle_id2": "product_count"}, inplace=True)
    #join order_user_id table and order_aisle_count table to see which order belongs to which user
    order_user_aisle = order_aisle_count.merge(order_user_id, on='order_id')
    #from order_user_aisle table, take out orderID and group by user_id, aisle_id
    user_aisle_id = order_user_aisle[['user_id','aisle_id','product_count']]
    user_aisle_count = user_aisle_id.groupby(['user_id','aisle_id']).agg({'product_count': 'sum'}).reset_index()
    # user_aisle_count right join user_order_count 
    user_aisle_order_count= user_aisle_count.merge(user_order_count, how='right',on='user_id')
    # final aisle grouping along with filtration of essential aisles
    new_user_aisle_order_count = user_aisle_order_count.merge(aisle_match, how='inner', on='aisle_id')
    new_user_aisle_order_count = new_user_aisle_order_count[['user_id','new_aisle_ID','product_count','order_count']]
    new_user_aisle_order_count = new_user_aisle_order_count.groupby(['user_id','new_aisle_ID']).agg({'product_count':'sum','order_count':'sum'}).reset_index()
    # divide product count by order count to get for aisle, how many products on average each customer would buy per order
    new_user_aisle_order_count['avg'] = new_user_aisle_order_count['product_count']/new_user_aisle_order_count['order_count']
    user_aisle_avgCount_annual = new_user_aisle_order_count[['user_id','new_aisle_ID','avg']]
    # get final data
    data = pd.pivot_table(user_aisle_avgCount_annual, values='avg', index=['user_id'],columns=['new_aisle_ID'], aggfunc=np.sum, fill_value=0)
    data.reset_index(inplace=True)
                                                                                                    
    # index match
    index_match = data['user_id']
    data.set_index('user_id', inplace=True)
    
    # Model prediction
    # apply principal component analysis to reduce the dimension from 69 to 34 dimensions
    pca = PCA(n_components=34)
    pca_34 = pca.fit_transform(data)
    pca_34_90percent = pd.DataFrame(pca_34)
                                                                                                     
    # load the model from disk and predict on the data
    filename = 'model/customer_segmentation_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_kmean = loaded_model.predict(pca_34_90percent)
    result = pca_34_90percent.copy(deep=True)
    result.reset_index(inplace=True)
    result.insert(1,'label',y_kmean)
    result = result.rename(columns = {'index':'X'})
    result = result[['X','label']]
    index_match = pd.DataFrame(index_match)
    index_match.reset_index(inplace=True)
    index_match = index_match.rename(columns = {'index':'X'})
    result = index_match.merge(result, on = 'X')
    result = result[['user_id','label']]
                                                                                                     
    return result


def rfc(filepath):
    orders = pd.read_csv(filepath + '/sample_order.csv')
    order_products = pd.read_csv(filepath + '/sample_order_products.csv')
    products = pd.read_csv(filepath + '/products.csv')
    orders['eval_set'] = orders['eval_set'].astype('category')

    def average_perc(x):
            order_reorder_prob = x.groupby('order_id')['reordered'].mean().to_frame('reorder_prob')
            return pd.Series({'reorder_item_ratio_per_order': statistics.mean(order_reorder_prob['reorder_prob'])}) 
        
    def preprocess():
        order_product = orders.merge(order_products, on='order_id', how='inner')

        user_total_orders = order_product.groupby('user_id').apply(lambda x: pd.Series({'total_orders': max(x['order_number'])})).reset_index()

        user_reorder = order_product.groupby('user_id').apply(lambda x: pd.Series({'reorder_ratio' : statistics.mean(x['reordered'])})).reset_index()

        order_product['days_since_prior_order'] = order_product['days_since_prior_order'].fillna(0)

        user_last_order = order_product.groupby('user_id').apply(lambda x: pd.Series({'avg_day_since_last_reorder' : statistics.mean(x['days_since_prior_order'])})).reset_index()
        
        average_reorder = order_product.groupby(['user_id']).apply(average_perc).reset_index()

        user = user_total_orders.merge(user_reorder, on='user_id', how='left')
        user = user.merge(user_last_order, on='user_id', how='left')
        user = user.merge(average_reorder, on='user_id', how='left')

        del user_reorder
        del user_last_order
        del average_reorder
        gc.collect()

        product_total_purchase = order_product.groupby('product_id').apply(lambda x: pd.Series({'total_purchase': len(x['order_id'])})).reset_index()
        product_reorder_ratio = order_product.groupby('product_id').apply(lambda x: pd.Series({'reorder_prob': statistics.mean(x['reordered'])})).reset_index()
        product = product_total_purchase.merge(product_reorder_ratio, on='product_id', how='left')
        del product_reorder_ratio
        del product_total_purchase

        gc.collect()

        total_bought = order_product.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('total_bought').reset_index()

        data = total_bought.merge(user, on='user_id', how='left')
        data = data.merge(product, on='product_id', how='left')
        data['reorder_prob'] = data['reorder_prob'].fillna(0)

        del user
        del product
        del total_bought

        data = data.set_index(['user_id', 'product_id'])
        return data

    rfc_model = pickle.load(open('model/rfc_model.sav', 'rb'))
    data = preprocess()
    pred = (rfc_model.predict_proba(data)[:,1] >= 0.30).astype(int)
    data['pred_reorder'] = pred
    data = data.reset_index()
    pred_user = data.groupby('user_id').apply(lambda x: list(x[x['pred_reorder'] == 1]['product_id'])).reset_index()
    products = products[['product_id', 'product_name']]
    result = {}
    for index, row in pred_user.iterrows():
        temp = []
        for i in row[0]:
            temp.append(products[products['product_id'] == i]['product_name'].array[0])
        result[row['user_id']] = temp

    return result

def returnRecommandItem(filepath, customercluster):
    apriori_output = pd.read_csv(filepath + 'model/output_Aprioir.csv')
    apriori_output = apriori_output[['first','second','label']]
    
    # read sample data
    aisle = pd.read_csv(filepath + '/aisles.csv')
    department = pd.read_csv(filepath + '/departments.csv')
    order_products = pd.read_csv(filepath + '/sample_order_products.csv')
    orders = pd.read_csv(filepath + '/sample_order.csv')
    products = pd.read_csv(filepath + '/products.csv')
    aisle_match = pd.read_csv(filepath + '/final_aisle_match.csv')
    
    order_user_id = orders[['order_id','user_id']]
    user_product_id=order_user_id.merge(order_products,left_on='order_id', right_on='order_id')
    user_product_id = user_product_id[['user_id','product_id']]
    user_product_label = customercluster.merge(user_product_id,left_on='user_id', right_on='user_id')


    user_product_dic={}
    for item in user_product_label.itertuples():
        if(user_product_dic.get(item[1])):
            user_product_dic[item[1]].append(item[3])
        else:
            user_product_dic[item[1]] = []
            user_product_dic[item[1]].append(item[3])
    predict_dic = {}
    for item in user_product_dic:
        predict_set=[]
        pids = user_product_dic[item]
        label = user_product_label[user_product_label['user_id']==item]['label'].values[0]
        apriori_output_part = apriori_output[apriori_output['label']==label]
        for item2 in apriori_output_part.itertuples():
            if(len(item2[1].strip('[]'))<7):
                if(int(item2[1].strip('[]')) in pids):
                    if(len(item2[2].strip('[]'))<7):
                        predict_set.append(int(item2[2].strip('[]')))
                    
        predict_dic[item] = predict_set
    for item in predict_dic:
        pro_arr = []
        for item2 in predict_dic[item]:
            pro_arr.append(products[products['product_id' ]== item2]['product_name'].values[0])
        predict_dic[item] = pro_arr
    return predict_dic


if __name__ == "__main__":
    folder_name = sys.argv[1]
    cluster = customerSegmentation(folder_name)
    association_result = returnRecommandItem(folder_name, cluster)
    rfc_result = rfc(folder_name)

    print("========= K-Means clustering result first 10 =========")
    print(cluster.head(10))
    print()
    print("========= Association rule result first 10 =========")
    for key in list(association_result)[:10]:
        print("user id: " , key, "--->", association_result[key])
    print()
    print("========= Random Forest Classifier Reorder result first 10 =========")
    for key in list(rfc_result)[:10]:
        print("user id: " , key, "--->", rfc_result[key])

    with open('association_rule_output.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in association_result.items():
           p_list = ', '.join(map(str, value))
           writer.writerow([key, p_list])

    with open('rfc_output.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in rfc_result.items():
           p_list = ', '.join(map(str, value))
           writer.writerow([key, p_list])