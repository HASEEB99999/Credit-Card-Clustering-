# Credit Card Customer Segmentation

## Unsupervised Learning Project - Week 4

---

## * Project Overview

This project applies unsupervised machine learning techniques to segment credit card customers based on their behavioral data. Using **K-Means** and **Hierarchical** clustering algorithms, we identify natural groupings within the customer base to enable targeted marketing strategies, effective risk management, and personalized customer services.

The analysis leverages behavioral data of approximately 9,000 active credit card holders over a six-month period, uncovering three distinct customer segments with unique characteristics and business implications.

---

## * Key Findings

### Optimal Number of Clusters: **3 Customer Segments Identified**

| Segment | Size | Percentage | Business Label |
|---------|------|------------|----------------|
| **Segment 0** | 4,161 | 46.5% | Low-Engagement Customers |
| **Segment 1** | 3,305 | 36.9% | High-Value Customers |
| **Segment 2** | 1,484 | 16.6% | Installment Spenders |

### Algorithm Recommendation
**K-Means clustering** is recommended for production use due to:
- Superior scalability with large datasets
- Better interpretability of cluster profiles
- Clear actionable insights for business stakeholders
- Consistent and stable results

### Business Impact
- * Enables personalized marketing strategies
- * Improves customer retention through targeted programs
- * Supports risk management through customer profiling
- * Optimizes resource allocation across customer segments

---

## * Dataset Information

### Source
- **Kaggle**: Credit Card Dataset for Clustering
- **Link**: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
- **Description**: Behavioral data of approximately 9,000 active credit card holders over 6 months

### Dataset Statistics
- **Rows**: 8,950
- **Columns**: 18
- **Time Period**: 6 months

### Features Overview

| Feature | Description | Data Type |
|---------|-------------|-----------|
| CUST_ID | Customer identifier | Object |
| BALANCE | Current account balance | Float |
| BALANCE_FREQUENCY | Frequency of balance updates | Float |
| PURCHASES | Total purchase amount | Float |
| ONEOFF_PURCHASES | One-time purchase amount | Float |
| INSTALLMENTS_PURCHASES | Installment purchase amount | Float |
| CASH_ADVANCE | Cash advance amount | Float |
| PURCHASES_FREQUENCY | Frequency of purchases | Float |
| ONEOFF_PURCHASES_FREQUENCY | Frequency of one-off purchases | Float |
| PURCHASES_INSTALLMENTS_FREQUENCY | Frequency of installment purchases | Float |
| CASH_ADVANCE_FREQUENCY | Frequency of cash advances | Float |
| CASH_ADVANCE_TRX | Number of cash advance transactions | Float |
| PURCHASES_TRX | Number of purchase transactions | Float |
| CREDIT_LIMIT | Credit limit | Float |
| PAYMENTS | Total payments made | Float |
| MINIMUM_PAYMENTS | Minimum payments made | Float |
| PRC_FULL_PAYMENT | Percentage of full payments | Float |
| TENURE | Customer tenure in months | Integer |

---

---

### Prerequisites

- **Python 3.8** or higher
- **pip** package manager
- **Git** (optional, for cloning repository)
- **Jupyter Notebook** or **Google Colab**

### Installation

#### Step 1: Clone the Repository


git clone https://github.com/HASEEB99999/Credit-Card-Clustering

#### Step 2: Create a Virtual Environment


# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

#### 📈 Methodology
1. Data Preprocessing
Data Cleaning

    Dropped identifier columns: CUST_ID (unique identifier)

    Handled missing values: Median imputation for MINIMUM_PAYMENTS (313 missing values, 3.50%)

    Verified data quality: No missing values remain after imputation

#### Feature Scaling

Why Scaling is Mandatory:

    Clustering algorithms use distance-based metrics

    Features with larger scales dominate the distance calculation

    StandardScaler ensures all features contribute equally

    Transforms features to mean=0 and standard deviation=1

#### python

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

#### 2. K-Means Clustering
Algorithm Overview

K-Means is a centroid-based algorithm that:

    Partitions data into K clusters

    Assigns each point to the nearest cluster centroid

    Iteratively updates centroids based on cluster memberships

    Minimizes within-cluster sum of squares (inertia)

#Finding Optimal Number of Clusters

#### Elbow Method:
Testing k values from 2 to 10, the elbow curve shows a sharp bend at k=3, after which the rate of decrease becomes more gradual.

#### Silhouette Scores:
While k=2 achieves the highest silhouette score (0.2898), k=3 provides better business interpretability with a reasonable score (0.2335).

Final Selection: k = 3 (Optimal)
k	Inertia	Silhouette Score
2	136,424.77	0.2898
3	118,232.19	0.2335
4	106,612.54	0.2074
5	98,654.44	0.1969
6	90,432.45	0.1946
7	84,142.29	0.1917
8	79,180.73	0.1856
9	74,332.70	0.1827
10	69,734.60	0.1821
3. Hierarchical Clustering
Algorithm Overview

#### Hierarchical clustering builds a hierarchy of clusters through:

    Starting with each point as its own cluster

    Merging the closest clusters iteratively

    Creating a dendrogram showing cluster relationships

#### Method Used: Agglomerative (bottom-up) with Ward's linkage

Sample Size: 300 rows (for dendrogram readability)
#### 4. Cluster Profiling
Analysis Performed

    Calculated mean of each feature per cluster

    Created heatmap visualization

    Interpreted clusters in business terms

    Developed actionable recommendations

#### 📊 Results Summary
Cluster Profiles
Cluster 0: Low-Engagement Customers

    Size: 4,161 customers (46.5%)

    Characteristics:

        Lowest average balance

        Lowest balance frequency

        Below-average purchase activity

        Lowest credit limits

        Low payment amounts

    Business Label: Low-Engagement Customers

    Recommended Actions:

        Offer cashback rewards to increase engagement

        Send targeted promotions based on spending patterns

        Consider increasing credit limits to encourage usage

        These are low-risk, low-value customers

# Cluster 1: High-Value Customers

    Size: 3,305 customers (36.9%)

    Characteristics:

        Highest average balance

        High balance frequency

        Above-average purchase activity

        Highest purchase frequency

        Highest payment amounts

    #### Business Label: High-Value Customers

    Recommended Actions:

        Implement premium loyalty programs

        Offer exclusive rewards and benefits

        Cross-sell other financial products

        Prioritize customer retention efforts

        These are high-profit, low-risk customers

# Cluster 2: Installment Spenders

    Size: 1,484 customers (16.6%)

    Characteristics:

        Above-average balance

        Highest balance frequency

        Highest installment purchases

        Highest credit limits

        High minimum payments

    Business Label: Installment Spenders

    #### Recommended Actions:

        Monitor credit risk closely

        Offer installment loan products

        Provide balance transfer options

        Send payment reminders to prevent defaults

        These are high-revenue but also high-risk customers

# Algorithm Comparison
Aspect	K-Means	Hierarchical

Scalability	✅ Excellent (O(n*k))	❌ Poor (O(n³))

Interpretability	✅ Clear centroids	⚠️ Dendrogram complex

Cluster Quality	✅ Well-separated	⚠️ Less distinct

Stability	✅ Consistent	⚠️ Sensitive to noise

Business Use	✅ Production-ready	⚠️ Exploratory only

Agreement Rate	-	48.67%
📈 Visualizations
1. Elbow Curve

Shows the optimal number of clusters (k=3) where the inertia decrease becomes gradual.
2. Silhouette Scores

Validates the cluster quality, confirming k=3 as a good choice.
3. Cluster Heatmap

Profiles each segment's characteristics with red (above average) and blue (below average) cells.
4. Dendrogram

Shows hierarchical relationships between clusters with a horizontal threshold line at the optimal cut.
#### 💡 Business Recommendations
##### Marketing Strategies

    High-Value Customers: Premium loyalty programs, exclusive offers, personalized communication

    Low-Engagement Customers: Engagement campaigns, cashback rewards, targeted promotions

    Installment Spenders: Product-specific offers, balance transfer options, credit line increases

##### Risk Management

    Monitor Installment Spenders: Implement early warning systems for potential defaults

    Track High-Value Customers: Identify any changes in spending patterns that might indicate financial distress

    Engage Low-Engagement Customers: Prevent account dormancy and potential churn

##### Product Development

    High-Value: Premium credit cards with enhanced benefits

    Low-Engagement: Simple, no-fee products to encourage usage

    Installment Spenders: Flexible installment plans, personalized loan products

#####  Technologies Used
Technology	Purpose	Version

Python	Programming language	3.8+

pandas	Data manipulation and analysis	1.3.0+

numpy	Numerical computing	1.21.0+

scikit-learn	Clustering algorithms, preprocessing	1.0.0+

scipy	Hierarchical clustering, dendrograms	1.7.0+

matplotlib	Data visualization	3.4.0+

seaborn	Statistical data visualization	0.11.0+

Jupyter	Interactive development environment	1.0.0+

📝 Key Code Examples

Data Preprocessing

python

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

##### Load data
df = pd.read_csv('data/DataSet(W4).csv')

##### Drop identifier column
df = df.drop('CUST_ID', axis=1)

##### Handle missing values
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())

##### Scale features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

K-Means Clustering
python

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

##### Find optimal k
inertia = []
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

##### Apply with optimal k (3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)

#### Hierarchical Clustering
python

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

#### Sample data
sample = df_scaled.sample(300, random_state=42)

#### Generate dendrogram
linkage_matrix = linkage(sample, method='ward')
dendrogram(linkage_matrix)

#### Apply clustering
agg = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = agg.fit_predict(sample)



# 🔍 Model Evaluation
Metrics Summary

Metric	Value

Optimal Number of Clusters	3

Silhouette Score (k=3)	0.2335

Inertia (k=3)	118,232.19

K-Means vs Hierarchical Agreement	48.67%

Total Customers Segmented	8,950

Features Used	17

