# CMPE255-InstaCart

## Project Proposal

#### Description
It seems like a lot of current market basket algorithms only focus on finding associations between products without taking into considerations of individual consumer preferences which at the end would result in product recommendations that consumers don’t really like. The objective of the project is to come up with a recommender system that is more tailored to individual consumer’s taste in order to enhance their online grocery shopping experience. First, the project will focus on dividing consumers into different categories (i.e. vegetarians, meat-lovers, etc.) based on their previous grocery records using clustering. Then market basket analysis will be applied to the grocery records to find association between products for each consumer category. Major assumption of the project is that consumers within the same category would be more willing to try the products that others, who have similar taste as them, would buy.

#### Methodology/Models
For consumer segmentation, individual products will be first grouped together into larger baskets based on their similarity and then customers will be divided into different categories based on their purchase history of those product baskets. Algorithms like K means clustering, Random Forest can be used for consumer division, Apriori could be applied for market basket analysis on each customer classification. 
It is expected that along with data preprocessing and additional understanding of the full data set, some of the methodologies/models might change.

For measurement metric, we will use precision, recall, f1-score to evaluate the model.
