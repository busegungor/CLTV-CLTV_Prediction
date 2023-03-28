import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
# import data
def import_csv(dataframe):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 500)
    pd.set_option("display.expand_frame_repr", False)
    df_ = pd.read_csv(dataframe)
    return df_

df_ = import_csv("/Users/busegungor/PycharmProjects/cltv_coffee_shop/coffee shop/201904 sales reciepts.csv")
df = df_.copy()

df.head()
# We want to catch outliers because of using probability model. So we want to get normal distribution for variables.
def outliers_threshold(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# After determined the up limit and low limit repressed.
def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outliers_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_threshold(df, "line_item_amount")
replace_with_threshold(df, "unit_price")
# according to customer taked groupby and transaction_id can be multiple times because they may have different item.
# so we want to know singular transaction id which can be taught nunique() function.
# if we want to know total price we can sum line_item_amount which means for each product's quantity multiply unit_price
cltv_c = df.groupby("customer_id").agg({"transaction_id": lambda x: x.nunique(),
                                        "line_item_amount": lambda x: x.sum()})

# and then we can change variable names as we know
cltv_c.columns = ["total_transaction", "total_price"]

# first of all we want to calculate average order value: Average Order Value = Total Price / Total Transaction
cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]
# second we have to calculate purchase frequency: Purchase Frequency = Total Transaction / Total Number of Customers
# cltv_c.shape method can figure out items, variables if we want to know variables we can look at first value.
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
# and then we want to know customer value: Customer Value = Average Order Value * Purchase Frequency
cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]
# if we calculate repeat rate we have to know total transaction is bigger than 1. we select bigger than 1 value
# with cltv_c[cltv_c["total_transaction"] > 1] and then .shape gives the size, again first value gives the items.
# Repeat Rate = Number of customers who made multiple purchases / Total Number of Customers
repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
# and then we can find the churn rate
churn_rate = 1 - repeat_rate
# if we want to know profit margin we can multiply total price with constant value which is determined by company.
cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10
# and finally we can find bottom formulation.
cltv_c["cltv"] = cltv_c["customer_value"] / churn_rate * cltv_c["profit_margin"]
# after found the cltv value can be sorted according to descending, we can seperate segments with qcut,
# worst values D segment, best values A segment.
cltv_c.sort_values(by="cltv", ascending=False)
cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])
# we can analyze the segments according to sum, mean values.
cltv_c.groupby("segment").agg({"sum", "mean", "count"})
# also save these values with csv file.
cltv_c.to_csv("cltv_c.csv")

# Funcitionalization
def calculate_cltv(dataframe):
    cltv_c = dataframe.groupby("customer_id").agg({"transaction_id": lambda x: x.nunique(),
                                                    "line_item_amount": lambda x: x.sum()})
    cltv_c.columns = ["total_transaction", "total_price"]
    cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
    cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10
    cltv_c["cltv"] = cltv_c["customer_value"] / churn_rate * cltv_c["profit_margin"]
    cltv_c.sort_values(by="cltv", ascending=False)
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])
    return cltv_c

calculate_cltv(df)

# Prepare Lifetime Data Structure

# recency: time from the last transaction. weekly ( customer in particular)
# T: how long before the analysis date the first purchase was made
# frequency: total number of repeat purchases (frequency > 1)
# monetary: average earnings per purchase

df["transaction_date"] = pd.to_datetime(df["transaction_date"])
today_date = dt.datetime(2019, 5, 1)
cltv_df = df.groupby("customer_id").agg({"transaction_date": [lambda date: (date.max() - date.min()).days,
                                                              lambda date: (today_date - date.min()).days],
                                         "transaction_id": lambda num: num.nunique(),
                                         "line_item_amount": lambda x: x.sum()})


cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
# this monetary for all earnings we want to know average earnings per purchase
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
# frequency: total number of repeat purchases (frequency > 1)
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]
# we want to know weekly date for all date variables.
cltv_df["T"] = cltv_df["T"] / 7
cltv_df["recency"] = cltv_df["recency"] / 7

# Establishment of BG-NBD Model : BetaGeoFitter() concrete a object with penalizer_coef arguments which means the
# penalty coefficient to be applied during the finding of the parameters. And model have to fit frequencyi recency
# and T values. As a result, according to the buy till you die principle, the transaction rates (**Buy**)
# vary according to each customer and the gamma is distributed for the whole audience (r, a).
# Dropout rates (**Till you die**) vary for each client and beta is spread out for the entire audience (a, b).
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])
# Out[58]: <lifetimes.BetaGeoFitter: fitted with 2237 subjects, a: 0.14, alpha: 0.74, b: 4.25, r: 2.11>

# Who are the 10 customers we expect the most to purchase in 1 week? 1 = 1 week
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)
# The conditional_expected_number_of_purchases_up_to_time can also be found with the predict function.
cltv_df["expected_purchase_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"])
# Who are the 10 customers we expect the most to purchase in 1 month? 4 = 4 weeks = 1 month
bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)



cltv_df["expected_purchase_1_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"])

bgf.predict(4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]).sum()
# The number of transactions expected by the company in a month. 16351.908539150718

# evaluating expected results

plot_period_transactions(bgf)
plt.show()

# Establishing GAMMA-GAMMA Model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])
# Out[70]: <lifetimes.GammaGammaFitter: fitted with 2237 subjects, p: 3.89, q: 2.85, v: 3.62>

ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])


# Calculation of CLTV with BG-NBD and GG Model
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3, # 3 aylık kaç aylık hesap isttiyorsun
                                   freq="W", # T'nin frekans bilgisi
                                   discount_rate=0.01)

cltv = cltv.reset_index()
cltv_final = cltv_df.merge(cltv, on="customer_id", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# As seen, when sorted from high to low according to clv, only expected average profit does not have a contribution to
# the value because even if expected average profit is high, it can be seen that it is lower below. If the customer with
# a regular average transaction capacity did not churn, the probability of purchase increases as the recency value
# increases. This is related to the bg-nbd theory (buy till you die). It can be seen here that only monetary,
# only frequency, or only customer age have not been sorted. There are many observations where the purchase frequency is
# low but the monetary value is high, or the purchase frequency is high but the monetary value is low. If we sorted
# by only one of these, another value would have been disregarded.**

# Make up Segmentation According to CLTV: After performing some calculations such as rule-based or RFM analysis before,
# we are doing segmentation processes here for the same reason because we already have rankings based on many values and
# we divide them into segments to easily deal with these individuals. However, it is possible to increase or decrease
# the segments depending on the distributions of these segments on other variables.
cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.sort_values(by="clv", ascending=False).head(50)
cltv_final.groupby("segment").agg({"sum", "mean", "count"})

# Since we calculated the average value each customer will bring to us, we also calculated the average income that all
# of our customers will leave to us. Therefore, we can determine how much we spend to find new customers
# if we know how much we spend. Questions like how should I proceed if the return that existing customers will bring in
# the next 6 months are so much can be asked.














