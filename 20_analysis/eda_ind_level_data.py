# INGEST, PRE-PROCESS, CONCATENATE AND CARRY OUT EDA FOR INDIVIDUAL LEVEL DATA
# 1. INGEST DATA

# import modules and packages
import pandas as pd
import numpy as np
import altair as alt
from scipy import stats
import statsmodels.api as sm
import scipy.stats
from matplotlib import pyplot as plt
import seaborn as sns

# ingest data sets
y04 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_clarissa/years04_05_.csv"
)
y05 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_clarissa/years05_06_.csv"
)
y06 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_clarissa/years06_07_.csv"
)
y07 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_john/years07_08_f.csv"
)
y08 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_john/years08_09_f.csv"
)
y10 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_john/years10_11_f.csv"
)
y11 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_john/years11_12_f.csv"
)
y12 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_rashaad/years12_13.csv"
)
y13 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_rashaad/years13_14.csv"
)
y14 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_rashaad/years14_15.csv"
)
y15 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_preet/years15_16_.csv"
)
y18 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_preet/years18_19_.csv"
)
y19 = pd.read_csv(
    "/Users/johnowusuduah/github/uds-2022-ids-701-team-3/10_data_cleaning/clean_preet/years19_20_.csv"
)


# 2. CLEAN DATA
# 2004 - data cleaning for consistency
# drop unecessary columns
y04 = y04.drop(["id"], axis=1)
# convert hhcode to non-decimal string
y04["hhcode"] = y04["hhcode"].astype("str")
y04["hhcode"] = y04["hhcode"].apply(lambda x: x[0:-2])
# make sure first letter in province is capital
y04["province"] = y04["province"].str.title()
# reindex columns for consistency across all data sets
y04i = y04.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y04i.sample(5)

# 2005 - data cleaning for consistency
# drop unecessary columns
y05 = y05.drop(["id"], axis=1)
# convert hhcode to non-decimal string
y05["hhcode"] = y05["hhcode"].astype("str")
y05["hhcode"] = y05["hhcode"].apply(lambda x: x[0:-2])
# reindex columns for consistency across all data sets
y05i = y05.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y05i.sample(5)

# 2006 - data cleaning for consistency
# drop unecessary columns
y06 = y06.drop(["id"], axis=1)
# convert hhcode to non-decimal string
y06["hhcode"] = y06["hhcode"].astype("str")
y06["hhcode"] = y06["hhcode"].apply(lambda x: x[0:-2])
# convert integer values of region variable to string labels
y06.loc[y06["region"] == 1, "region"] = "urban"
y06.loc[y06["region"] == 2, "region"] = "rural"
# convert "integer-like" values of educational enrollment status to string labels
y06.loc[y06["currently_enrolled"] == "1.0", "currently_enrolled"] = "yes"
y06.loc[y06["currently_enrolled"] == "2.0", "currently_enrolled"] = "no"
# convert integer values of ever admitted variable to string labels
y06.loc[y06["ever_admitted"] == 1, "ever_admitted"] = "yes"
y06.loc[y06["ever_admitted"] == 2, "ever_admitted"] = "no"
# reindex columns
y06i = y06.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y06i.sample(5)

# 2007 - data cleaning for consistency
# drop unecessary columns
y07 = y07.drop(["Unnamed: 0"], axis=1)
# convert integer value of region variable to string label
y07.loc[y07["region"] == "3", "region"] = "urban"
# reindex columns
y07i = y07.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y07i.sample(5)

# 2008 - data cleaning for consistency
# drop unecessary columns
y08 = y08.drop(["Unnamed: 0"], axis=1)
# reindex columns
y08i = y08.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y08i.sample(5)

# 2010 - data cleaning for consistency
# drop unecessary columns
y10 = y10.drop(["Unnamed: 0"], axis=1)
y10i = y10.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y10i.sample(5)

# 2011 - data cleaning for consistency
# drop unecessary columns
y11 = y11.drop(["Unnamed: 0"], axis=1)
y11i = y11.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y11i.sample(5)

# 2012 data cleaning for consistency
# drop unnecessary columns
y12 = y12.drop(["Unnamed: 0"], axis=1)
# convert hhcode to non-decimal string
y12["hhcode"] = y12["hhcode"].astype("str")
y12["hhcode"] = y12["hhcode"].apply(lambda x: x[0:-2])
# reindex columns
y12i = y12.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y12i.sample(5)

# 2013 data cleaning for consistency
# data cleaning for consistency
# drop unnecessary columns
y13 = y13.drop(["Unnamed: 0"], axis=1)
# convert hhcode to non-decimal string
y13["hhcode"] = y13["hhcode"].astype("str")
y13["hhcode"] = y13["hhcode"].apply(lambda x: x[0:-2])
# rename stratum to subprovince for consistency
y13.rename(columns={"stratum": "subprovince"}, inplace=True)
# reindex columns
y13i = y13.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y13i.sample(5)

# 2014 data cleaning for consistency
# data cleaning for consistency
# drop unnecessary columns
y14 = y14.drop(["Unnamed: 0"], axis=1)
# convert hhcode to non-decimal string
y14["hhcode"] = y14["hhcode"].astype("str")
y14["hhcode"] = y14["hhcode"].apply(lambda x: x[0:-2])
# reindex columns
y14i = y14.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y14i.sample(5)

# 2015 data cleaning for consistency
# data cleaning for consistency
# drop unnecessary columns
y15 = y15.drop(["id"], axis=1)
# convert hhcode to non-decimal string
y15["hhcode"] = y15["hhcode"].astype("str")
y15["hhcode"] = y15["hhcode"].apply(lambda x: x[0:-2])
# rename Year for consistency
y15.rename(columns={"Year": "year"}, inplace=True)
# reindex columns
y15i = y15.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y15i.sample(5)

# 2018 data cleaning for consistency
# data cleaning for consistency
# drop unnecessary columns
y18 = y18.drop(["id"], axis=1)
# convert hhcode to non-decimal string
y18["hhcode"] = y18["hhcode"].astype("str")
y18["hhcode"] = y18["hhcode"].apply(lambda x: x[0:-2])
# rename Year for consistency
y18.rename(columns={"Year": "year"}, inplace=True)
y18i = y18.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y18i.sample(5)

# 2019 data cleaning for consistency
# data cleaning for consistency
# drop unnecessary columns
y19 = y19.drop(["id"], axis=1)
# convert hhcode to non-decimal string
y19["hhcode"] = y19["hhcode"].astype("str")
y19["hhcode"] = y19["hhcode"].apply(lambda x: x[0:-2])
# rename Year for consistency
y19.rename(columns={"Year": "year"}, inplace=True)
y19i = y19.reindex(
    columns=[
        "hhcode",
        "age",
        "idc",
        "sex",
        "marital_status",
        "ever_admitted",
        "currently_enrolled",
        "region",
        "subprovince code",
        "province",
        "subprovince",
        "year",
    ],
    copy=True,
)
# preview data
y19i.sample(5)


# 3. CONCATENATE INDIVIDUAL DATA TO ONE DATA SET
# concatenate data sets
df = pd.concat(
    [y04i, y05i, y06i, y07i, y08i, y10i, y11i, y12i, y13i, y14i, y15i, y18i, y19i],
    axis=0,
)

# correct mispelt values of province columns
df["province"] = df["province"].replace("Nwfp", "KPK")
df["province"] = df["province"].replace("NWFP", "KPK")
df["province"] = df["province"].replace("NWFP ", "KPK")
df["province"] = df["province"].replace("NWFP ", "NWFP")
df["province"] = df["province"].replace("Punjab ", "Punjab")
df["province"] = df["province"].replace("K.P.K", "KPK")
df["province"] = df["province"].replace("Islamabad", "Punjab")

# preview a sample of the data
df.sample(5)

# check data types
df.info()

# check the distribution of numeric variables
df.describe()

# 4. PRELIMINARY EDA
# extract the number of observations in the data set
f"There are {df.shape[0]} observations in the data set."

# check value counts of sex
df.sex.value_counts().reset_index()

# check value counts of educational enrollment
df.currently_enrolled.value_counts().reset_index()

# check value counts of ever admitted variable
df.ever_admitted.value_counts().reset_index()

# check value counts of region variable
df.region.value_counts().reset_index()

# EDA FOR OVERALL DATA SET
# Pre-process Data for EDA
# convert string categorical variables to integer labels
df_1 = df.copy()
# convert sex
df_1.loc[df_1["sex"] == "male", "sex"] = 0
df_1.loc[df_1["sex"] == "female", "sex"] = 1
# convert educational enrollment
df_1.loc[df_1["currently_enrolled"] == "no", "currently_enrolled"] = 0
df_1.loc[df_1["currently_enrolled"] == "yes", "currently_enrolled"] = 1
# convert ever admitted
df_1.loc[df_1["ever_admitted"] == "no", "ever_admitted"] = 0
df_1.loc[df_1["ever_admitted"] == "yes", "ever_admitted"] = 1
# convert region
df_1.loc[df_1["region"] == "rural", "region"] = 0
df_1.loc[df_1["region"] == "urban", "region"] = 1

# preview data
df_1.sample(5)


# 5. FINAL EDA
# For Difference-in-Difference Analysis,
# check for balance across the treatment arm, ie. sex,
# for age, ever_admitted and region
for i in ["age", "ever_admitted", "region"]:
    female = df_1.loc[df_1.sex == 1, i].mean()
    male = df_1.loc[df_1.sex == 0, i].mean()
    pvalue = stats.ttest_ind(
        df_1.loc[df_1.sex == 1, i].values,
        df_1.loc[df_1.sex == 0, i].values,
    ).pvalue
    print(f"For {i}, the mean for females in the survey is {female:.3f},")
    print(f"the mean for males in the survey is {male:.3f},")
    print(f"and the p-value for this difference is {pvalue:.3f}")
    print("\n")

f"We see that age, whether a student has been admitted in an educational institution, and region are statistically significantly different across both male and \
females in the data set. This justifies our use of difference-in-difference to estimate causal inference"


# crosstab of treatment variable versus response variable
pd.crosstab(df["currently_enrolled"], df["sex"], margins=True)

# normalization for all variables have been done by rows so fractions
# are in terms of
# normalized crosstab of treatment variable versus response variable
pd.crosstab(df["currently_enrolled"], df["sex"], margins=True, normalize="index")

# crosstab of treatment variable versus region
pd.crosstab(df["region"], df["sex"], margins=True)

# normalized crosstab of treatment variable versus region
pd.crosstab(df["region"], df["sex"], margins=True, normalize="index")

# crosstab of treatment variable versus ever_admitted
pd.crosstab(df["ever_admitted"], df["sex"], margins=True)

# normalized crosstab of treatment variable versus ever_admitted
pd.crosstab(df["ever_admitted"], df["sex"], margins=True, normalize="index")

# crosstab of treatment variable versus marital status
pd.crosstab(df["marital_status"], df["sex"], margins=True)

# normalized crosstab of treatment variable versus marital status
# these values vary so much that we shall not test for whether the
# difference is statistically different
pd.crosstab(df["marital_status"], df["sex"], margins=True, normalize="index")

# crosstab of treatment variable province
pd.crosstab(df["province"], df["sex"], margins=True)

# normalized crosstab of treatment variable province
pd.crosstab(df["province"], df["sex"], margins=True, normalize="index")

# crosstab of treatment variable year
pd.crosstab(df["year"], df["sex"], margins=True)

# normalized crosstab of treatment variable year
pd.crosstab(df["year"], df["sex"], margins=True, normalize="index")
