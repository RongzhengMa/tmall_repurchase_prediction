import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import seaborn as sns
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('data/train_format1.csv')
user_info = pd.read_csv('data/user_info_format1.csv')
user_log = pd.read_csv('data/user_log_format1.csv')

user_info['age_range'].replace(0.0,np.nan,inplace=True)
user_info['gender'].replace(2.0,np.nan,inplace=True)
user_info['age_range'].replace(np.nan,-1,inplace=True)
user_info['gender'].replace(np.nan,-1,inplace=True)


fig = plt.figure(figsize = (10, 6))
x = np.array(["NULL","<18","18-24","25-29","30-34","35-39","40-49",">=50"])
y = np.array([user_info[user_info['age_range'] == -1]['age_range'].count(),
             user_info[user_info['age_range'] == 1]['age_range'].count(),
             user_info[user_info['age_range'] == 2]['age_range'].count(),
             user_info[user_info['age_range'] == 3]['age_range'].count(),
             user_info[user_info['age_range'] == 4]['age_range'].count(),
             user_info[user_info['age_range'] == 5]['age_range'].count(),
             user_info[user_info['age_range'] == 6]['age_range'].count(),
             user_info[user_info['age_range'] == 7]['age_range'].count() + user_info[user_info['age_range'] == 8]['age_range'].count()])
plt.bar(x,y,label='Count')
plt.xlabel("Age Range")
plt.ylabel("Count")
plt.title('Age Distribution (By Range)')


user_info['age_range'] = user_info['age_range'].fillna(-1).astype(int)
existing_values = sorted(user_info['age_range'].unique())
plt.figure(figsize=(10, 6))
sns.countplot(x='age_range', order=existing_values, data=user_info)
plt.xlabel("Age Groups")
plt.ylabel("Count")
plt.title("Age Distribution (By Group)")
plt.show()


user_info['gender'] = user_info['gender'].fillna(-1).astype(int)
existing_values = sorted(user_info['gender'].unique())
plt.figure(figsize=(10, 6))
sns.countplot(x='gender', order=existing_values, data=user_info)
plt.xlabel("Gender Groups")
plt.ylabel("Count")
plt.title("Gender Distribution")
plt.show()


user_info['age_range'] = user_info['age_range'].fillna(-1).astype(int)
user_info['gender'] = user_info['gender'].fillna(-1).astype(int)
existing_values = sorted(user_info['age_range'].unique())
plt.figure(figsize=(10, 6))
sns.countplot(x='age_range', hue='gender', order=existing_values, data=user_info)
plt.xlabel("Age Range")
plt.ylabel("Count")
plt.title("Age and Gender Distribution")
plt.legend(title="Gender")
plt.show()


user_info['age_range'].replace(-1,np.nan,inplace=True)
user_info['gender'].replace(-1,np.nan,inplace=True)

user_log['time_stamp'].hist(bins = 9)
plt.title("Time Stamp Distribution") 
plt.xlabel("Time Stamp(MMDD)") 
plt.ylabel("Frequency") 
plt.show()


user_log['action_type'] = user_log['action_type'].fillna(-1).astype(int)
existing_values = sorted(user_log['action_type'].unique())
plt.figure(figsize=(10,6))
sns.countplot(x='action_type', order=existing_values, data=user_log)
plt.xlabel("Action Type")
plt.ylabel("Count")
plt.title("User Action Distribution")
plt.show()


df_train = pd.merge(df_train,user_info,on="user_id",how="left")

total_logs_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"]]).count().reset_index()[["user_id","seller_id","item_id"]]
total_logs_temp.rename(columns={"seller_id":"merchant_id","item_id":"total_logs"},inplace=True)
df_train = pd.merge(df_train,total_logs_temp,on=["user_id","merchant_id"],how="left")


unique_item_ids_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["item_id"]]).count().reset_index()[["user_id","seller_id","item_id"]]
unique_item_ids_temp1 = unique_item_ids_temp.groupby([unique_item_ids_temp["user_id"],unique_item_ids_temp["seller_id"]]).count().reset_index()
unique_item_ids_temp1.rename(columns={"seller_id":"merchant_id","item_id":"unique_item_ids"},inplace=True)
df_train = pd.merge(df_train,unique_item_ids_temp1,on=["user_id","merchant_id"],how="left")


categories_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["cat_id"]]).count().reset_index()[["user_id","seller_id","cat_id"]]
categories_temp1 = categories_temp.groupby([categories_temp["user_id"],categories_temp["seller_id"]]).count().reset_index()
categories_temp1.rename(columns={"seller_id":"merchant_id","cat_id":"categories"},inplace=True)
df_train = pd.merge(df_train,categories_temp1,on=["user_id","merchant_id"],how="left")


browse_days_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["time_stamp"]]).count().reset_index()[["user_id","seller_id","time_stamp"]]
browse_days_temp1 = browse_days_temp.groupby([browse_days_temp["user_id"],browse_days_temp["seller_id"]]).count().reset_index()
browse_days_temp1.rename(columns={"seller_id":"merchant_id","time_stamp":"browse_days"},inplace=True)
df_train = pd.merge(df_train,browse_days_temp1,on=["user_id","merchant_id"],how="left")


one_clicks_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["action_type"]]).count().reset_index()[["user_id","seller_id","action_type","item_id"]]
one_clicks_temp.rename(columns={"seller_id":"merchant_id","item_id":"times"},inplace=True)
one_clicks_temp["one_clicks"] = one_clicks_temp["action_type"] == 0
one_clicks_temp["one_clicks"] = one_clicks_temp["one_clicks"] * one_clicks_temp["times"]
one_clicks_temp["shopping_carts"] = one_clicks_temp["action_type"] == 1
one_clicks_temp["shopping_carts"] = one_clicks_temp["shopping_carts"] * one_clicks_temp["times"]
one_clicks_temp["purchase_times"] = one_clicks_temp["action_type"] == 2
one_clicks_temp["purchase_times"] = one_clicks_temp["purchase_times"] * one_clicks_temp["times"]
one_clicks_temp["favourite_times"] = one_clicks_temp["action_type"] == 3
one_clicks_temp["favourite_times"] = one_clicks_temp["favourite_times"] * one_clicks_temp["times"]
four_features = one_clicks_temp.groupby([one_clicks_temp["user_id"],one_clicks_temp["merchant_id"]]).sum().reset_index()
four_features = four_features.drop(["action_type","times"], axis=1)
df_train = pd.merge(df_train,four_features,on=["user_id","merchant_id"],how="left")


df_train.isnull().sum(axis=0)
df_train = df_train.ffill()
df_train.info()

colnm = df_train.columns.tolist()
plt.figure(figsize = (5, 4))
color = sns.color_palette()

df_train[colnm[5]].hist(range=[0,80],bins = 80,color = color[1])
plt.xlabel(colnm[5],fontsize = 12)
plt.ylabel('Count')

df_train[colnm[6]].hist(range=[0,40],bins = 40,color = color[1])
plt.xlabel(colnm[6],fontsize = 12)
plt.ylabel('Count')

df_train[colnm[7]].hist(range=[0,10],bins = 10,color = color[1])
plt.xlabel(colnm[7],fontsize = 12)
plt.ylabel('Count')

df_train[colnm[8]].hist(range=[0,10],bins = 10,color = color[1])
plt.xlabel(colnm[8],fontsize = 12)
plt.ylabel('Count')

df_train[colnm[9]].hist(range=[0,50],bins = 50,color = color[1])
plt.xlabel(colnm[9],fontsize = 12)
plt.ylabel('One_clicks Count')

df_train[colnm[10]].hist(range=[0,3],bins = 3,color = color[1])
plt.xlabel(colnm[10],fontsize = 12)
plt.ylabel('Count')

df_train[colnm[11]].hist(range=[0,6],bins = 7,color = color[1])
plt.xlabel(colnm[11],fontsize = 12)
plt.ylabel("Count")

df_train[colnm[12]].hist(range=[0,6],bins = 6,color = color[1])
plt.xlabel(colnm[12],fontsize = 12)
plt.ylabel("Count")


sns.set_style("dark")
plt.figure(figsize = (10,8))
colnm = df_train.columns.tolist()[2:13]
mcorr = df_train[colnm].corr()
mask = np.zeros_like(mcorr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True,fmt='0.2f')


train_df, test_df = train_test_split(df_train, test_size=0.2, random_state=42)
train_df.to_csv("data/train_set.csv", index=False)
test_df.to_csv("data/test_set.csv", index=False)




