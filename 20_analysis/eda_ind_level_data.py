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


# 3. CONCATENATE INDIVIDUAL DATA TO ONE DATA SET AND FILTER AGES BETWEEN 4 AND 10
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

# filter of children between the ages of 4 and 10
df = df[(df["age"] > 4) & (df["age"] < 10)].copy()

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
df.sex.value_counts(normalize=True).reset_index()

# check value counts of educational enrollment
df.currently_enrolled.value_counts(normalize=True).reset_index()

# check value counts of ever admitted variable
df.ever_admitted.value_counts().reset_index()

# check value counts of region variable
df.region.value_counts().reset_index()

# check value counts of marital status variable
df.marital_status.value_counts().reset_index()


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


# ### 5. Final EDA (treatment variable --> sex)

# check balance for 2019 as an example
df_04 = df_1[df_1.year == 2019]


# 5. FINAL EDA
# For Difference-in-Difference Analysis,
# check for balance across the treatment arm, ie. sex,
# for age, ever_admitted and region
# age_mean = []
# ever_admitted_mean = []
# region_mean = []
for i in ["age", "ever_admitted", "region"]:
    female = df_04.loc[df_04.sex == 1, i].mean()
    male = df_04.loc[df_04.sex == 0, i].mean()
    pvalue = stats.ttest_ind(
        df_04.loc[df_04.sex == 1, i].values,
        df_04.loc[df_04.sex == 0, i].values,
    ).pvalue
    print(f"For {i}, the mean for females in the survey is {female:.3f},")
    print(f"the mean for males in the survey is {male:.3f},")
    print(f"and the p-value for this difference is {pvalue:.3f}")
    print("\n")

f"We see that age, whether a student has been admitted in an educational institution, and region are statistically significantly different across both male and \
females for a majority of the years in the data set. This would invalidate causal inference analysis on the on enrollment between men and women. We could match men and women \
for years where there are baseline difference but we would be reducing the statistical power of our analysis. So we decided to investigate the causal impact of the taliban attacks \
on women in rural areas controlled by the taliban compared to women in rural areas not controlled by the taliban."


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

df_1.groupby(["year", "sex"])["currently_enrolled"].mean().reset_index()

# normalized crosstab of treatment variable versus ever_admitted
pd.crosstab(df["ever_admitted"], df["sex"], margins=True, normalize="index")

# crosstab of treatment variable versus marital status
pd.crosstab(df["marital_status"], df["region"], margins=True)

# normalized crosstab of treatment variable versus marital status
# these values vary so much that we shall not test for whether the
# difference is statistically different
pd.crosstab(df["marital_status"], df["region"], margins=True, normalize="columns")

# crosstab of treatment variable province
pd.crosstab(df["province"], df["sex"], margins=True)

# normalized crosstab of treatment variable province
pd.crosstab(df["province"], df["sex"], margins=True, normalize="index")

# crosstab of treatment variable year
pd.crosstab(df["year"], df["sex"], margins=True)

# normalized crosstab of treatment variable year
pd.crosstab(df["year"], df["sex"], margins=True, normalize="index")


# ### 6. FINAL EDA (TREATMENT VARIABLE --> taliban controlled areas versus non-taliban controlled in rural areas)
# a. Pre-processing of subprovince name ---> Ensure Consistency of Subprovince Names
# Replaces:
df_1["subprovince"] = df_1["subprovince"].replace("Ättock", "Attock")
df_1["subprovince"] = df_1["subprovince"].replace("Abbottabad", "Abbotabad")
df_1["subprovince"] = df_1["subprovince"].replace("Bahawalnagar", "Bahawal Nagar")
df_1["subprovince"] = df_1["subprovince"].replace("Bahawalnager", "Bahawal Nagar")

df_1["subprovince"] = df_1["subprovince"].replace("Bhakhar", "Bhakkar")
df_1["subprovince"] = df_1["subprovince"].replace("Bhakar", "Bhakkar")
df_1["subprovince"] = df_1["subprovince"].replace("Baddin", "Badin")

df_1["subprovince"] = df_1["subprovince"].replace("Barkhen", "Barkhan")

df_1["subprovince"] = df_1["subprovince"].replace("Bhawalpur", "Bahawalpur")
df_1["subprovince"] = df_1["subprovince"].replace("Bolan/ Kachhi", "Bolan/Kachhi")
df_1["subprovince"] = df_1["subprovince"].replace("Bolan/Kachhi", "Bolan/Kachhi")
df_1["subprovince"] = df_1["subprovince"].replace("Bolan/ kachhi", "Bolan/Kachhi")
df_1["subprovince"] = df_1["subprovince"].replace("Bolan/Kachni", "Bolan/Kachhi")

df_1["subprovince"] = df_1["subprovince"].replace("Bonair", "Buner")
df_1["subprovince"] = df_1["subprovince"].replace("Bunair", "Buner")


df_1["subprovince"] = df_1["subprovince"].replace("Chaghi", "Chagai")
df_1["subprovince"] = df_1["subprovince"].replace("Chaghai", "Chagai")
df_1["subprovince"] = df_1["subprovince"].replace("Chaghi", "Chagai")
df_1["subprovince"] = df_1["subprovince"].replace("Charsada", "Charsadda")
df_1["subprovince"] = df_1["subprovince"].replace("D. G. Khan", "D.G. Khan")
df_1["subprovince"] = df_1["subprovince"].replace("Dera Ghazi Khan", "D.G. Khan")
df_1["subprovince"] = df_1["subprovince"].replace("D. g. khan", "D.G. Khan")
df_1["subprovince"] = df_1["subprovince"].replace("D.G.Khan", "D.G. Khan")
df_1["subprovince"] = df_1["subprovince"].replace("D.g khan", "D.G. Khan")


df_1["subprovince"] = df_1["subprovince"].replace("D.I.Khan", "D.I. Khan")
df_1["subprovince"] = df_1["subprovince"].replace("D. i. khan", "D.I. Khan")
df_1["subprovince"] = df_1["subprovince"].replace("D. I. Khan", "D.I. Khan")
df_1["subprovince"] = df_1["subprovince"].replace("Dera Ismail Khan", "D.I. Khan")
df_1["subprovince"] = df_1["subprovince"].replace("D.i.khan", "D.I. Khan")

df_1["subprovince"] = df_1["subprovince"].replace("Dera bugti", "Dera Bugti")
df_1["subprovince"] = df_1["subprovince"].replace("Deara Bughti", "Dera Bugti")
df_1["subprovince"] = df_1["subprovince"].replace("Dera Bughti", "Dera Bugti")

df_1["subprovince"] = df_1["subprovince"].replace("Gujranwala", "Gujaranwala")
df_1["subprovince"] = df_1["subprovince"].replace("Gawadar", "Gwadar")

df_1["subprovince"] = df_1["subprovince"].replace("Haifzabad", "Hafizabad")
df_1["subprovince"] = df_1["subprovince"].replace("Hafaizabad", "Hafizabad")
df_1["subprovince"] = df_1["subprovince"].replace("Hzara", "Hazara")
df_1["subprovince"] = df_1["subprovince"].replace("Pak Pattain", "Pakpattan")
df_1["subprovince"] = df_1["subprovince"].replace("Pakpaten", "Pakpattan")
df_1["subprovince"] = df_1["subprovince"].replace("Pakpatan", "Pakpattan")
df_1["subprovince"] = df_1["subprovince"].replace("Pakpatten", "Pakpattan")

df_1["subprovince"] = df_1["subprovince"].replace("pishine", "Pishin")
df_1["subprovince"] = df_1["subprovince"].replace("Pishine", "Pishin")
df_1["subprovince"] = df_1["subprovince"].replace("Pashin", "Pishin")
df_1["subprovince"] = df_1["subprovince"].replace("Pershawar", "Peshawar")
df_1["subprovince"] = df_1["subprovince"].replace("Qillah Abdullah", "Qilla Abdullah")
df_1["subprovince"] = df_1["subprovince"].replace("QillahAbdullah", "Qilla Abdullah")
df_1["subprovince"] = df_1["subprovince"].replace("Killa Abdullah", "Qilla Abdullah")
df_1["subprovince"] = df_1["subprovince"].replace("Qilla abdullah", "Qilla Abdullah")

df_1["subprovince"] = df_1["subprovince"].replace("Qillah Saifullah", "Qilla Saifullah")
df_1["subprovince"] = df_1["subprovince"].replace(
    "Qillah Salifullah", "Qilla Saifullah"
)
df_1["subprovince"] = df_1["subprovince"].replace("QillahSaifullah", "Qilla Saifullah")
df_1["subprovince"] = df_1["subprovince"].replace("Killa Saifullah", "Qilla Saifullah")
df_1["subprovince"] = df_1["subprovince"].replace("Qilla saifullah", "Qilla Saifullah")

df_1["subprovince"] = df_1["subprovince"].replace("Quetta (Div)", "Quetta")
df_1["subprovince"] = df_1["subprovince"].replace("R.Y.Khan", "Rahim Yar Khan")
df_1["subprovince"] = df_1["subprovince"].replace("Rahim yar khan", "Rahim Yar Khan")
df_1["subprovince"] = df_1["subprovince"].replace("Rajaanpur", "Rajanpur")
df_1["subprovince"] = df_1["subprovince"].replace(
    "Shaheed Banazir Abad", "Shaheed Benazir Abad"
)
df_1["subprovince"] = df_1["subprovince"].replace(
    "Shaheed Benazirabad", "Shaheed Benazir Abad"
)
df_1["subprovince"] = df_1["subprovince"].replace(
    "Shaheed benazir abad", "Shaheed Benazir Abad"
)
df_1["subprovince"] = df_1["subprovince"].replace("Sheerani", "Sherani")
df_1["subprovince"] = df_1["subprovince"].replace("Sibbi (Div)", "Sibbi")
df_1["subprovince"] = df_1["subprovince"].replace("Sibi", "Sibbi")
df_1["subprovince"] = df_1["subprovince"].replace("Sijawal", "Sujawal")
df_1["subprovince"] = df_1["subprovince"].replace("Sukkar", "Sukkur")
df_1["subprovince"] = df_1["subprovince"].replace("T.T. Singh", "Toba Tek Singh")
df_1["subprovince"] = df_1["subprovince"].replace("T.T.Singh", "Toba Tek Singh")
df_1["subprovince"] = df_1["subprovince"].replace("T.t. singh", "Toba Tek Singh")
df_1["subprovince"] = df_1["subprovince"].replace("T.t singh", "Toba Tek Singh")
df_1["subprovince"] = df_1["subprovince"].replace("Sarghodha", "Sargodha")
df_1["subprovince"] = df_1["subprovince"].replace("Sheani", "Sherani")
df_1["subprovince"] = df_1["subprovince"].replace("Shikarpur", "Shiokarpur")

df_1["subprovince"] = df_1["subprovince"].replace(
    "Tando Muhammad", "Tando Muhammad Khan"
)
df_1["subprovince"] = df_1["subprovince"].replace(
    "Tando Muhd Khan", "Tando Muhammad Khan"
)
df_1["subprovince"] = df_1["subprovince"].replace(
    "Tando mohammad khan", "Tando Muhammad Khan"
)
df_1["subprovince"] = df_1["subprovince"].replace(
    "Tando mohd khan", "Tando Muhammad Khan"
)

df_1["subprovince"] = df_1["subprovince"].replace("TandoAllah Yar", "Tando Allahyar")
df_1["subprovince"] = df_1["subprovince"].replace("Tando Allah Yar", "Tando Allahyar")
df_1["subprovince"] = df_1["subprovince"].replace("Tando allah yar", "Tando Allahyar")
df_1["subprovince"] = df_1["subprovince"].replace("Tor Garh", "Torghar")
df_1["subprovince"] = df_1["subprovince"].replace("Tor ghar", "Torghar")
df_1["subprovince"] = df_1["subprovince"].replace("Torgarh", "Torghar")

df_1["subprovince"] = df_1["subprovince"].replace("Umer kot", "Umer Kot")
df_1["subprovince"] = df_1["subprovince"].replace("Ümer kot", "Umer Kot")
df_1["subprovince"] = df_1["subprovince"].replace("Upper dir", "Upper Dir")
df_1["subprovince"] = df_1["subprovince"].replace("UpperDir", "Upper Dir")
df_1["subprovince"] = df_1["subprovince"].replace("Zhob (Div)", "Zhob")


# %%
# more pre-processing of subprovince names
df_1["subprovince"] = df_1["subprovince"].replace("Jaccobabad", "Jacobabad")
df_1["subprovince"] = df_1["subprovince"].replace("Jaffarabad", "Jafarabad")
df_1["subprovince"] = df_1["subprovince"].replace("Jafrabad", "Jafarabad")
df_1["subprovince"] = df_1["subprovince"].replace("Jhal magsi", "Jhal Magsi")
df_1["subprovince"] = df_1["subprovince"].replace("JhalMagsi", "Jhal Magsi")
df_1["subprovince"] = df_1["subprovince"].replace("Jehlum", "Jhelum")

df_1["subprovince"] = df_1["subprovince"].replace("Kachhi/ Bolan", "Kachhi")
df_1["subprovince"] = df_1["subprovince"].replace("Bolan/kachhi", "Kachhi")
df_1["subprovince"] = df_1["subprovince"].replace("Bolan/Kachhi", "Kachhi")

df_1["subprovince"] = df_1["subprovince"].replace("Kalat (Div)", "Kalat")
df_1["subprovince"] = df_1["subprovince"].replace("Karachi Central", "Karachi")
df_1["subprovince"] = df_1["subprovince"].replace("Karachi East", "Karachi")
df_1["subprovince"] = df_1["subprovince"].replace("Karachi Malir", "Karachi")
df_1["subprovince"] = df_1["subprovince"].replace("Karachi South", "Karachi")
df_1["subprovince"] = df_1["subprovince"].replace("Karachi West", "Karachi")
df_1["subprovince"] = df_1["subprovince"].replace("Kashmore", "Kashmor")
df_1["subprovince"] = df_1["subprovince"].replace("Kech", "Kech/Turbat")
df_1["subprovince"] = df_1["subprovince"].replace("Ketch/Turbat", "Kech/Turbat")
df_1["subprovince"] = df_1["subprovince"].replace("Keych/turbat", "Kech/Turbat")

df_1["subprovince"] = df_1["subprovince"].replace("Killa abdullah", "Qilla Saifullah")
df_1["subprovince"] = df_1["subprovince"].replace("Killa saifullah", "Killa Saifullah")
df_1["subprovince"] = df_1["subprovince"].replace("Lakki marwat", "Lakki Marwat")
df_1["subprovince"] = df_1["subprovince"].replace("LakkiMarwat", "Lakki Marwat")
df_1["subprovince"] = df_1["subprovince"].replace("Lasbella", "Lasbela")
df_1["subprovince"] = df_1["subprovince"].replace("Lasbilla", "Lasbela")
df_1["subprovince"] = df_1["subprovince"].replace("Layyah", "Layya")
df_1["subprovince"] = df_1["subprovince"].replace("Lodhrean", "Lodhran")

df_1["subprovince"] = df_1["subprovince"].replace("Lower dir", "Lower Dir")
df_1["subprovince"] = df_1["subprovince"].replace("LowerDir", "Lower Dir")
df_1["subprovince"] = df_1["subprovince"].replace("Makran (Div)", "Makran")
df_1["subprovince"] = df_1["subprovince"].replace("Malakand Protected", "Malakand")
df_1["subprovince"] = df_1["subprovince"].replace("Malakand Protected Area", "Malakand")
df_1["subprovince"] = df_1["subprovince"].replace("Mandi Bahuddin", "Mandi Bahauddin")
df_1["subprovince"] = df_1["subprovince"].replace("Mandi bahauddin", "Mandi Bahauddin")
df_1["subprovince"] = df_1["subprovince"].replace("Manshera", "Mansehra")
df_1["subprovince"] = df_1["subprovince"].replace("Mir pur Khas", "Mir Pur Khas")
df_1["subprovince"] = df_1["subprovince"].replace("MirPurKhas", "Mir Pur Khas")
df_1["subprovince"] = df_1["subprovince"].replace("Mirpur Khas", "Mir Pur Khas")
df_1["subprovince"] = df_1["subprovince"].replace("Mirpur khas", "Mir Pur Khas")
df_1["subprovince"] = df_1["subprovince"].replace("Mir pur khas", "Mir Pur Khas")
df_1["subprovince"] = df_1["subprovince"].replace("M ianwali", "Mianwali")
df_1["subprovince"] = df_1["subprovince"].replace("Mitiari", "Matiari")

df_1["subprovince"] = df_1["subprovince"].replace("MusaKhel", "Musa Khel")
df_1["subprovince"] = df_1["subprovince"].replace("Musa khel", "Musa Khel")
df_1["subprovince"] = df_1["subprovince"].replace("Musa", "Musa Khel")
df_1["subprovince"] = df_1["subprovince"].replace("Musakhel", "Musa Khel")
df_1["subprovince"] = df_1["subprovince"].replace("Muraffar Garh", "Muzaffar Garh")
df_1["subprovince"] = df_1["subprovince"].replace("Muzaffar garh", "Muzaffar Garh")
df_1["subprovince"] = df_1["subprovince"].replace("Muzaffargarh", "Muzaffar Garh")
df_1["subprovince"] = df_1["subprovince"].replace("Nankana Sahi", "Nankana Sahib")
df_1["subprovince"] = df_1["subprovince"].replace("Nankana sahib", "Nankana Sahib")
df_1["subprovince"] = df_1["subprovince"].replace("Naseerabad (Div)", "Nasirabad")
df_1["subprovince"] = df_1["subprovince"].replace("Nasirabad/ Tamboo", "Nasirabad")
df_1["subprovince"] = df_1["subprovince"].replace("Nasirabad/ tamboo", "Nasirabad")
df_1["subprovince"] = df_1["subprovince"].replace(
    "Naushahro feroze", "Naushahro Feroze"
)
df_1["subprovince"] = df_1["subprovince"].replace("Nowshero Feroze", "Naushahro Feroze")
df_1["subprovince"] = df_1["subprovince"].replace("Nowshero Freoze", "Naushahro Feroze")
df_1["subprovince"] = df_1["subprovince"].replace("Nowsehra", "Nowshera")
df_1["subprovince"] = df_1["subprovince"].replace("Nawabsha", "Nawabshah")
df_1["subprovince"] = df_1["subprovince"].replace("Nowshero feroze", "Naushahro Feroze")
df_1["subprovince"] = df_1["subprovince"].replace("Nauski", "Nushki")

df_1["subprovince"] = df_1["subprovince"].replace("Umer kot", "Umer Kot")
df_1["subprovince"] = df_1["subprovince"].replace("Upper dir", "Upper Dir")
df_1["subprovince"] = df_1["subprovince"].replace("UpperDir", "Upper Dir")
df_1["subprovince"] = df_1["subprovince"].replace("Zhob (Div)", "Zhob")


# subset for rural areas and for women
df_r = df_1[(df_1["region"] == 0) & (df_1["sex"] == 1)].copy()

# add an indicator variable for whether an area is a taliban controlled area or not
taliabn_dominance = [
    "South Waziristan",
    "North Waziristan",
    "Orakzai",
    "Kurram",
    "Khyber",
    "Mohmand",
    "Bajur",
    "Darra Adamkhel",
    "Swat",
    "Upper Dir",
    "Lower Dir",
    "Bannu",
    "Lakki Marwat",
    "Tank",
    "Peshawar",
    "Dera Ismail Khan",
    "Mardan",
    "Charsadda",
    "Kohat",
]
df_r.loc[df_r["subprovince"].isin(taliabn_dominance), "taliban"] = 1
df_r.loc[~df_r["subprovince"].isin(taliabn_dominance), "taliban"] = 0
df_r.taliban = df_r.taliban.astype("int")
# preview value counts of observations in taliban controlled areas versus
df_r.taliban.value_counts()

# preview data set
df_r.sample(5)

# check for balance in the treatment group which comprises of taliban controlled groups and control group
# which comprises of areas not controlled by the taliban
# check for balance across the treatment arm, ie. taliban,
# for age, ever_admitted and region
for i in ["age"]:
    taliban = df_r.loc[df_r.taliban == 1, i].mean()
    non_taliban = df_r.loc[df_r.taliban == 0, i].mean()
    pvalue = stats.ttest_ind(
        df_r.loc[df_r.taliban == 1, i].values,
        df_r.loc[df_r.taliban == 0, i].values,
    ).pvalue
    print(
        f"For {i}, the mean for taliban controlled areas in the survey is {taliban:.3f},"
    )
    print(
        f"the mean for non-taliban controlled areas in the survey is {non_taliban:.3f},"
    )
    print(f"and the p-value for this difference is {pvalue:.3f}")
    print("\n")

# check whether composition of intervention and comparision groups is stable
# for repeated cross-sectional design
ctab = pd.crosstab(df_r["year"], df_r["taliban"], margins=True, normalize="index")
ctab

chi2, p, dof, expected = scipy.stats.chi2_contingency(ctab.values)
f" The p-value between the treatment and control groups across cross-sections of the \
data is {p:0.3f}. So the composition of treatment and control groups is stable across \
cross-sections"

# so the difference is not statistically different across the
# two groups for

f"We see that age is not statistically significantly different between women in rural taliban controlled areas versus \
women in rural areas not controlled by the taliban. This shows that there are no baseline differences between these two groups and \
our approach to determine the causal inference of the terrorist attacks on the two groups using difference-in-difference is justified."
