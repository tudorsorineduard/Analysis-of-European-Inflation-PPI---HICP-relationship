import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#Loading the datasets

hicp_unprocessed = pd.read_csv('prc_hicp_minr__custom_21750876_linear.csv', low_memory=False)
ppi_unprocessed  = pd.read_csv('sts_inpp_m__custom_21751016_linear.csv', low_memory=False)

print("HICP ok:", hicp_unprocessed.shape)
print("PPI ok:", ppi_unprocessed.shape)

#Both datasets contain 2 measurement types mixed together. Separating for different uses
#Index values -> clustering and visualizations
#Rate of change values -> Granger, CCF and forecasting

#HICP
hicp_index = hicp_unprocessed[hicp_unprocessed['unit'] == 'Index, 2025=100'][['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
hicp_rate = hicp_unprocessed[hicp_unprocessed['unit'] == 'Annual rate of change'][['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()

#PPI

ppi_rate = ppi_unprocessed[ppi_unprocessed['unit'] == 'Percentage change compared to same period in previous year'][['geo', 'nace_r2', 'TIME_PERIOD', 'OBS_VALUE']].copy()
ppi_index = ppi_unprocessed[ppi_unprocessed['unit'] == 'Index, 2021=100'][['geo', 'nace_r2', 'TIME_PERIOD', 'OBS_VALUE']].copy()

print("HICP index shape:", hicp_index.shape)
print("HICP rate shape:", hicp_rate.shape)
print("PPI rate shape:", ppi_rate.shape)
print("PPI index shape:", ppi_index.shape)

#Long to wide format for clustering and time series models
#For PPI a separate wide dataframe for each MIG category

def to_wide(df, index_col, col_col, val_col):
    wide = df.pivot(index=index_col, columns=col_col, values=val_col)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()

hicp_index_wide = to_wide(hicp_index, 'TIME_PERIOD', 'geo', 'OBS_VALUE')
hicp_rate_wide = to_wide(hicp_rate, 'TIME_PERIOD', 'geo', 'OBS_VALUE')

ppi_categories = ppi_rate['nace_r2'].unique()
ppi_wide = {}
ppi_index_wide = {}

for cat in ppi_categories:
    ppi_wide[cat] = to_wide(ppi_rate[ppi_rate['nace_r2'] == cat], 'TIME_PERIOD', 'geo', 'OBS_VALUE')
    ppi_index_wide[cat] = to_wide(ppi_index[ppi_index['nace_r2'] == cat], 'TIME_PERIOD', 'geo', 'OBS_VALUE')

print(f"HICP index wide shape: {hicp_index_wide.shape}")
print(f"HICP rate wide shape:  {hicp_rate_wide.shape}")

print("\nPPI wide shapes category:")
for cat in ppi_categories:
    print(f"  - {cat:<35} | index shape: {ppi_index_wide[cat].shape} | rate shape: {ppi_wide[cat].shape}")

#Checking for missing values

def check_missing(df, name):
    total = df.isna().sum().sum()
    print(f"{name}: {total} missing values" if total > 0 else f"{name}: no missing values")

check_missing(hicp_index_wide, "HICP Index")
check_missing(hicp_rate_wide, "HICP Rate")
for cat, df in ppi_wide.items():
    check_missing(df, f"PPI Rate - {cat}")
for cat, df in ppi_index_wide.items():
    check_missing(df, f"PPI Index - {cat}")

print("Missing values by country")
missing = hicp_index_wide.isna().sum()
print(f"HICP index: {missing[missing > 0].to_string() or 'no missing values'}")

missing = hicp_rate_wide.isna().sum()
print(f"HICP rate: {missing[missing > 0].to_string() or 'no missing values'}")

for cat, df in ppi_wide.items():
    missing = df.isna().sum()
    print(f"PPI rate - {cat}: {missing[missing > 0].to_string() or 'no missing values'}")

for cat, df in ppi_index_wide.items():
    missing = df.isna().sum()
    print(f"PPI index - {cat}: {missing[missing > 0].to_string() or 'no missing values'}")

#Handling mising values and synchronizing time for both datasets
#Removing countries with large data gaps that cannot be corrected/filled
#if a country doesn't have missing values in most categories
#removing just those with missing values
#Ireland has no PPI data in all categories for 136 months
#decided to remove it from the study
hicp_index_wide = hicp_index_wide.drop(columns=['Ireland'], errors='ignore')
hicp_rate_wide = hicp_rate_wide.drop(columns=['Ireland'], errors='ignore')

for cat in ppi_wide:
    ppi_wide[cat] = ppi_wide[cat].drop(columns=['Ireland'], errors='ignore')
    ppi_index_wide[cat] = ppi_index_wide[cat].drop(columns=['Ireland'], errors='ignore')

#Finland has no PPI MIG-Energy data available - all 244 missing values
ppi_wide['MIG - energy'] = ppi_wide['MIG - energy'].drop(columns=['Finland'], errors='ignore')
ppi_index_wide['MIG - energy'] = ppi_index_wide['MIG - energy'].drop(columns=['Finland'], errors='ignore')

#Luxembourg has no PPI MIG-Durable consumer goods data available - all 244 missing values
ppi_wide['MIG - durable consumer goods'] = ppi_wide['MIG - durable consumer goods'].drop(columns=['Luxembourg'], errors='ignore')
ppi_index_wide['MIG - durable consumer goods'] = ppi_index_wide['MIG - durable consumer goods'].drop(columns=['Luxembourg'], errors='ignore')

#Forward fill to handle small gaps with missing values

hicp_index_wide = hicp_index_wide.ffill()
hicp_rate_wide = hicp_rate_wide.ffill()

for cat in ppi_wide:
    ppi_wide[cat] = ppi_wide[cat].ffill()
    ppi_index_wide[cat] = ppi_index_wide[cat].ffill()

#Making HICP match PPI's latest available period - April 2026

hicp_index_wide = hicp_index_wide.loc[:'2026-04']
hicp_rate_wide  = hicp_rate_wide.loc[:'2026-04']

#Checking for missing after ffill and dropping the month May from HICP

check_missing(hicp_index_wide, "HICP Index")
check_missing(hicp_rate_wide, "HICP Rate")
for cat, df in ppi_wide.items():
    check_missing(df, f"PPI Rate - {cat}")
for cat, df in ppi_index_wide.items():
    check_missing(df, f"PPI Index - {cat}")

#Stationarity testing using the Augmented Dickey-Fuller (ADF) test
#p-value < 0.05 means stationary, p-value >= 0.05 means non-stationary

def adf_test(df, panel_name):
    stationary_count = 0
    non_stationary_count = 0

    for country in df.columns:
        series_clean = df[country].dropna()
        if len(series_clean) > 10:
            result = adfuller(series_clean)
            p_value = result[1]

            if p_value < 0.05:
                stationary_count += 1
            else:
                non_stationary_count += 1

    print(f"Panel: {panel_name:<55} -> Stationary: {stationary_count:<2} | Non-Stationary: {non_stationary_count:<2}")
    return stationary_count, non_stationary_count

adf_test(hicp_index_wide, "HICP indexes")
adf_test(hicp_rate_wide, "HICP rates")

for cat in ppi_wide:
    adf_test(ppi_index_wide[cat], f"PPI index for - {cat}")
    adf_test(ppi_wide[cat], f"PPI rate for  - {cat}")

#Non-stationary series need differencing prior to using them in Granger and forecasting models
#As the rates are non-stationary, applying differencing
#which removes the trend and makes the series stationary
#Indexes are not to be differenced, because they are only used for clustering and visualizations

hicp_rate_diff = hicp_rate_wide.diff().dropna()
ppi_diff = {}
for cat, df in ppi_wide.items():
    ppi_diff[cat] = df.diff().dropna()

adf_test(hicp_rate_diff, "HICP rate differenced")
for cat in ppi_diff:
    adf_test(ppi_diff[cat], f"PPI rate differenced - {cat}")

#Scaling the data
#StandardScaler for clustering to group by how countries' inflation moved through time
#instead of inflation values in absolute terms

scaler_standard = StandardScaler()
hicp_index_scaled = pd.DataFrame(
    scaler_standard.fit_transform(hicp_index_wide.T).T,
    index=hicp_index_wide.index,
    columns=hicp_index_wide.columns
)
print("\n HICP index scaled")
print(hicp_index_scaled.head(3))

#Figure 1 - HICP Annual Rate of Change per country

fig, ax = plt.subplots(figsize=(14, 7))

for country in hicp_rate_wide.columns:
    ax.plot(hicp_rate_wide.index, hicp_rate_wide[country], linewidth=0.9, alpha=0.7, label=country)

ax.set_title('HICP Annual Rate of Change (2006-2026)')
ax.set_xlabel('Year')
ax.set_ylabel('Annual Rate of Change (%)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hicp_trend_per_country.png', dpi=150, bbox_inches='tight')
plt.show()

#Figure 2 - Heatmap of HICP Annual Rate of Change per country per year

hicp_yearly = hicp_rate_wide.copy()
hicp_yearly['year'] = hicp_yearly.index.year
hicp_yearly_avg = hicp_yearly.groupby('year').mean().T

fig, ax = plt.subplots(figsize=(16, 10))
colors_list = ['#1a7a1a', '#5cb85c', '#f0e68c', '#e6a817', '#e85d04', '#9b0000']
positions = [0.0, 0.1, 0.2, 0.35, 0.6, 1.0]
cmap_custom = LinearSegmentedColormap.from_list('custom', list(zip(positions, colors_list)))
sns.heatmap(hicp_yearly_avg, cmap=cmap_custom, vmin=0, vmax=20,
            linewidths=0.3, ax=ax, cbar_kws={'label': 'Annual inflation (%)'})

ax.set_title('HICP Annual Inflation Rate by Country and Year (2006-2026)')
ax.set_xlabel('Year')
ax.set_ylabel('Country')
plt.tight_layout()
plt.savefig('hicp_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

#Figure 3 - PPI categories and HICP EU average 2006-2026 EU26

hicp_eu_avg = hicp_rate_wide.mean(axis=1)

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(hicp_eu_avg.index, hicp_eu_avg, label='HICP', color='black', linewidth=2)

colors = ['red', 'blue', 'green', 'orange', 'purple']
for (cat, df), color in zip(ppi_wide.items(), colors):
    eu_avg = df.mean(axis=1)
    ax.plot(eu_avg.index, eu_avg, label=cat, color=color, linewidth=0.9, alpha=0.7)

ax.set_title('HICP vs PPI Categories - EU Average (2006-2026)')
ax.set_xlabel('Year')
ax.set_ylabel('Annual Rate of Change (%)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hicp_vs_ppi_categories.png', dpi=150, bbox_inches='tight')
plt.show()

#Descriptive statistics for HICP and PPI

print("HICP Descriptive Statistics")
print(hicp_rate_wide.describe().T[['mean', 'std', 'min', 'max']].round(2))

print("\n PPI EU Average Descriptive Statistics")
for cat, df in ppi_wide.items():
    print(f"\n{cat}:")
    print(df.mean(axis=1).describe().round(2))

#Clustering

print("\n Clustering \n")

#Flipping countries from columns to rows for K-Means clustering
#Standardized monthly index to columns, being the features

clustering_data = hicp_index_scaled.T
print(f"Clustering data shape - countries and months: {clustering_data.shape}")

#Determining the right number of clusters

wcss = []
silhouette = []

#Checking between 2 and 6 clusters
k_range = range(2, 7)

for k in k_range:
    kmeans_check = KMeans(n_clusters=k, n_init=20, random_state=42)
    kmeans_check.fit(clustering_data)
    wcss.append(kmeans_check.inertia_)
    score = silhouette_score(clustering_data, kmeans_check.labels_)
    silhouette.append(score)
    print(f"K = {k} | WCSS: {kmeans_check.inertia_:.2f} | Silhouette Score: {score:.4f}")

#Figure 4 - Graphic plot for Elbow method and Silhouette Score

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'black'
ax1.set_xlabel('Number of clusters (K)', fontweight='bold')
ax1.set_ylabel('WCSS', color=color, fontweight='bold')
ax1.plot(k_range, wcss, marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
ax2.plot(k_range, silhouette, marker='s', color=color, linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('K-Means Evaluation with Elbow method and Silhouette Score', pad=15, fontweight='bold')
fig.tight_layout()
plt.savefig('k_means_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

#Established K=3 as the optimal number of clusters

optimal_k = 3

kmeans_final = KMeans(n_clusters=optimal_k, n_init=30, random_state=42)
cluster_labels = kmeans_final.fit_predict(clustering_data)

country_clusters = pd.Series(cluster_labels, index=clustering_data.index)
cluster_0 = []
cluster_1 = []
cluster_2 = []

for country, label in country_clusters.items():
    if label == 0:
        cluster_0.append(country)
    elif label == 1:
        cluster_1.append(country)
    elif label == 2:
        cluster_2.append(country)

cluster_list = [cluster_0, cluster_1, cluster_2]

for c in range(optimal_k):
    print(f"\n Cluster {c}  -> Contains {len(cluster_list[c])} countries:")
    print(", ".join(cluster_list[c]))

#Figure 5 - Implementing Principal Component Analysis for cluster visualizing

pca = PCA(n_components=2, random_state=42)
cluster_pca = pca.fit_transform(clustering_data)

pc1_variance = pca.explained_variance_ratio_[0] * 100
pc2_variance = pca.explained_variance_ratio_[1] * 100

print(f"PC1 explains {pc1_variance:.2f}% of the variance.")
print(f"PC2 explains {pc2_variance:.2f}% of the variance.")

pca_df = pd.DataFrame(data=cluster_pca, columns=['PC1', 'PC2'], index=clustering_data.index)
pca_df['Cluster'] = country_clusters.map({
    0: 'Cluster 0 - Stable',
    1: 'Cluster 1 - Very volatile',
    2: 'Cluster 2 - Intermediate'
})

plt.figure(figsize=(11, 8))

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Cluster',
    palette={
        'Cluster 0 - Stable': 'green',
        'Cluster 1 - Very volatile': 'red',
        'Cluster 2 - Intermediate': 'orange'
    },
    s=120,
    edgecolor='black',
    alpha=0.85,
    linewidth=0.8
)

for country in pca_df.index:
    plt.annotate(
        country,
        (pca_df.loc[country, 'PC1'], pca_df.loc[country, 'PC2']),
        textcoords="offset points",
        xytext=(0, 8),
        ha='center',
        fontsize=9,
        fontweight='bold'
    )

plt.title('Cluster visualisation using PCA', fontsize=13, fontweight='bold', pad=15)
plt.xlabel(f'Principal Component 1 ({pc1_variance:.1f}% Variance Explained)', fontweight='bold', labelpad=10)
plt.ylabel(f'Principal Component 2 ({pc2_variance:.1f}% Variance Explained)', fontweight='bold', labelpad=10)

plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='black')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.grid(True, linestyle=':', alpha=0.5)

plt.autoscale()
plt.tight_layout()
plt.savefig('hicp_pca_scatter_plot.png', dpi=150, bbox_inches='tight')
plt.show()

#Figure 6 - Annual Inflation Rate per cluster - Timeseries

fig, ax = plt.subplots(figsize=(14, 7))
colors = ['green', 'red', 'orange']

for c in range(optimal_k):
    cluster_data_ts = hicp_rate_wide[cluster_list[c]]
    cluster_mean = cluster_data_ts.mean(axis=1)
    cluster_max = cluster_data_ts.max(axis=1)
    cluster_min = cluster_data_ts.min(axis=1)

    ax.fill_between(cluster_mean.index, cluster_min, cluster_max, alpha=0.1, color=colors[c])
    ax.plot(cluster_mean.index, cluster_mean, linewidth=2, color=colors[c], label=f'Cluster {c} - mean')
    ax.plot(cluster_mean.index, cluster_max, linewidth=0.8, color=colors[c], alpha=0.5)

ax.set_title('HICP Inflation per Cluster - Mean and Range')
ax.set_xlabel('Year')
ax.set_ylabel('Annual Rate of Change (%)')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()

#Table with descriptive statistics for clusters

country_stats = hicp_rate_wide.describe().T[['mean', 'std', 'min', 'max']].round(2)
country_stats['Cluster'] = country_clusters

print(country_stats.sort_values('Cluster').to_string())
print("\n Cluster Averages")
print(country_stats.groupby('Cluster')[['mean', 'std', 'min', 'max']].mean().round(2).to_string())

#Granger causality - PPI - HICP - per cluster and EU wide
#Testing both directions to identify supply side (PPI->HICP) or demand-pull (HICP->PPI) dominance
#Using max_lag=4, testing 4 months backward

def run_granger(y, x, label, max_lag=4):
    data = pd.concat([y, x], axis=1).dropna()
    try:
        result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        best_lag = 1
        min_p = 1.0

        for lag in result:
            p_val = result[lag][0]['ssr_ftest'][1]
            if p_val < min_p:
                min_p = p_val
                best_lag = lag

        if min_p < 0.05:
            print(f"  {label:<50} p={min_p:.4f} is significant at {best_lag} months")
        else:
            print(f"  {label:<50} p={min_p:.4f} is not significant")

    except Exception as e:
        print(f"  {label:<50} error is {str(e)}")

#EU wide

print("\n EU wide Granger Causality test")
hicp_eu = hicp_rate_diff.mean(axis=1)

for cat, df in ppi_diff.items():
    ppi_eu = df.mean(axis=1)
    print(f"\n{cat}")
    run_granger(hicp_eu, ppi_eu, label="PPI -> HICP")
    run_granger(ppi_eu, hicp_eu, label="HICP -> PPI")

#Per Cluster
#Some countries are missing from their respective clusters, due to missing data in certain PPI categories

print("\n Cluster Granger Causality")

for c in range(optimal_k):
    print(f"\n Cluster {c}")
    hicp_cluster = hicp_rate_diff[cluster_list[c]].mean(axis=1)

    for cat, df in ppi_diff.items():
        countries = [col for col in df.columns if col in cluster_list[c]]
        if not countries:
            continue
        ppi_cluster = df[countries].mean(axis=1)
        print(f"\n  {cat}")
        run_granger(hicp_cluster, ppi_cluster, label="  PPI -> HICP")
        run_granger(ppi_cluster, hicp_cluster, label="  HICP -> PPI")

#Cross Correlation Function - to identify how many months PPI movement preceeds that of HICP
#Results used to construct the lag features for calculating Feature Importance using RF and for forecasting

print("\n Cross Correlation")

optimal_lags = {}
lag_datasets = {}

print("\n EU wide")
hicp_eu = hicp_rate_diff.mean(axis=1)
optimal_lags['EU'] = {}
eu_features = pd.DataFrame({'Target_HICP': hicp_eu})

for cat in ppi_diff:
    df = ppi_diff[cat]
    ppi_eu = df.mean(axis=1)

    ccf_values = ccf(ppi_eu, hicp_eu, nlags=5)

    best_lag = 1
    best_corr = 0
    for lag in range(1, 5):
        if abs(ccf_values[lag]) > abs(best_corr):
            best_corr = ccf_values[lag]
            best_lag = lag

    optimal_lags['EU'][cat] = best_lag
    print(f"  {cat:<40} lag: {best_lag} months | r={best_corr:.4f}")

    for l in range(1, 5):
        eu_features[f"{cat}_lag_{l}"] = ppi_eu.shift(l)

for l in range(1, 5):
    eu_features[f"HICP_lag_{l}"] = hicp_eu.shift(l)

lag_datasets['EU'] = eu_features.dropna()
print(f"  EU dataset shape: {lag_datasets['EU'].shape}")


for c in range(optimal_k):
    print(f"\n Cluster {c}")

    hicp_cluster = hicp_rate_diff[cluster_list[c]].mean(axis=1)
    cluster_features = pd.DataFrame({'Target_HICP': hicp_cluster})
    optimal_lags[c] = {}

    for cat in ppi_diff:
        df = ppi_diff[cat]
        countries = []
        for col in df.columns:
            if col in cluster_list[c]:
                countries.append(col)
        if not countries:
            continue

        ppi_cluster = df[countries].mean(axis=1)
        ccf_values = ccf(ppi_cluster, hicp_cluster, nlags=5)

        best_lag = 1
        best_corr = 0
        for lag in range(1, 5):
            if abs(ccf_values[lag]) > abs(best_corr):
                best_corr = ccf_values[lag]
                best_lag = lag

        optimal_lags[c][cat] = best_lag
        print(f"  {cat:<40} lag: {best_lag} months | r={best_corr:.4f}")

        for l in range(1, 5):
            cluster_features[f"{cat}_lag_{l}"] = ppi_cluster.shift(l)

    for l in range(1, 5):
        cluster_features[f"HICP_lag_{l}"] = hicp_cluster.shift(l)

    lag_datasets[c] = cluster_features.dropna()
    print(f"  Dataset shape: {lag_datasets[c].shape}")

#Random Forest Feature Importance per cluster, using optimal lag identified with CCF
#Two measurements used for predictor importance:

#Mean Decrease in Impurity with Residual Sum of Squares -
#verifies by how much PPI categories reduce prediction error across all splits

#Permutation Importance -
#verifies accuracy drops by shuffling feature's values

print("\n Random Forest Feature Importance per cluster")

rf_perm_results = {}
rf_mdi_results = {}

for c in range(optimal_k):
    print(f"\n Cluster {c}")

    X = lag_datasets[c].drop(columns=['Target_HICP'])
    y = lag_datasets[c]['Target_HICP']

    # Training Random Forest with OOB
    rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=42)
    rf.fit(X, y)
    print(f"  OOB R² Score: {rf.oob_score_:.4f}")

    #MDI per PPI category using optimal lag
    mdi_importances = rf.feature_importances_

    print(f"\n  PPI Categories - MDI):")
    mdi_category = {}
    for cat in ppi_diff:
        optimal_lag = optimal_lags[c][cat]
        col = f"{cat}_lag_{optimal_lag}"
        if col in X.columns:
            idx = list(X.columns).index(col)
            mdi_category[cat] = mdi_importances[idx]

    sorted_mdi = sorted(mdi_category.items(), key=lambda x: x[1], reverse=True)
    for cat, importance in sorted_mdi:
        print(f"  {cat:<40} MDI importance: {importance:.4f}")

    #Permutation per category with optimal lag
    perm_result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

    print(f"\n  PPI Category - Permutation Importance")
    category_importance = {}

    for cat in ppi_diff:
        optimal_lag = optimal_lags[c][cat]
        col = f"{cat}_lag_{optimal_lag}"
        if col in X.columns:
            idx = list(X.columns).index(col)
            category_importance[cat] = perm_result.importances_mean[idx]

    sorted_cat = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    for cat, importance in sorted_cat:
        print(f"  {cat:<40} Permutation importance: {importance:.4f}")

    rf_perm_results[c] = category_importance
    rf_mdi_results[c] = mdi_category

#Figure 7 - Random Forest feature importance per cluster with MDI and Permutation Importance

fig, axes = plt.subplots(2, optimal_k, figsize=(16, 10))

for c in range(optimal_k):

        #MDI
        ax1 = axes[0][c]
        sorted_mdi = sorted(rf_mdi_results[c].items(), key=lambda x: x[1])
        cat_mdi = [x[0].replace('MIG - ', '') for x in sorted_mdi]
        values_mdi = [x[1] for x in sorted_mdi]
        ax1.barh(cat_mdi, values_mdi, color='royalblue')
        ax1.set_title(f'Cluster {c} - MDI', fontweight='bold')
        ax1.set_xlabel('MDI Importance')
        ax1.grid(True, alpha=0.3)

        #Permutation Importance
        ax2 = axes[1][c]
        sorted_perm = sorted(rf_perm_results[c].items(), key=lambda x: x[1])
        cat_perm = [x[0].replace('MIG - ', '') for x in sorted_perm]
        values_perm = [x[1] for x in sorted_perm]
        ax2.barh(cat_perm, values_perm, color='firebrick')
        ax2.set_title(f'Cluster {c} - Permutation Importance', fontweight='bold')
        ax2.set_xlabel('Permutation Importance')
        ax2.grid(True, alpha=0.3)

plt.suptitle('RF Feature Importance per cluster - MDI & Permutation Importance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

#ARIMA forecasting HICP rates EU wide

print("\n ARIMA")

hicp_target = lag_datasets['EU']['Target_HICP']

split = int(len(hicp_target) * 0.8)
train = hicp_target[:split]
test = hicp_target[split:]

print(f" Train size: {len(train)} months")
print(f" Test size: {len(test)} months")

#Checking for best p, d, q using AIC
print("\n Checking for best ARIMA parameters")
best_aic = np.inf
best_order = None

for p in range(0, 4):
    for d in range(0,1):
        for q in range(0, 4):
            try:
                model = ARIMA(train, order=(p, d, q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
            except:
                continue

print(f" Best ARIMA order: {best_order} | AIC: {best_aic:.2f}")

#Training with the best parameters
#However using rolling one step forecast or walk forward validation
#Model predicts the next month and after each prediction, the real known value
#is added to its history, meaning the model will forecast knowing the real past valeus

history = list(train)
arima_predictions_diff = []

for t in range(len(test)):
    model = ARIMA(history, order=best_order)
    result = model.fit()
    pred = result.forecast(steps=1)[0]
    arima_predictions_diff.append(pred)
    history.append(test.iloc[t])

arima_predictions_diff = pd.Series(arima_predictions_diff, index=test.index)

#Transforming ARIMA results from differenced to original scale
#For comparison with real HICP rates and meaningful MAE and RMSE interpretation
#Each prediction is added to the real value of the previous month

hicp_rate_original = hicp_rate_wide.mean(axis=1)
arima_predictions_original = []

for idx in range(len(test)):
    prev_date = test.index[idx] - pd.DateOffset(months=1)
    real_prev_value = hicp_rate_original.loc[prev_date]
    arima_predictions_original.append(real_prev_value + arima_predictions_diff.iloc[idx])

arima_predictions_original = pd.Series(arima_predictions_original, index=test.index)

#Values in the original scale
train_original = hicp_rate_original.loc[train.index]
test_original = hicp_rate_original.loc[test.index]

#Evaluation on original scale
arima_mae = mean_absolute_error(test_original, arima_predictions_original)
arima_rmse = np.sqrt(mean_squared_error(test_original, arima_predictions_original))

print(f"\n  ARIMA{best_order} Results:")
print(f"  MAE:  {arima_mae:.4f}")
print(f"  RMSE: {arima_rmse:.4f}")

#Figure 8 - ARIMA rolling one step forecast compared to actual HICP rate
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_original.index, train_original, color='black', label='Train', linewidth=1)
ax.plot(test_original.index, test_original, color='blue', label='Actual', linewidth=1.5)
ax.plot(test_original.index, arima_predictions_original, color='red', label=f'ARIMA forecast', linewidth=1.5, linestyle='--')
ax.set_title(f'ARIMA rolling one step forecast & actual HICP rate')
ax.set_xlabel('Date')
ax.set_ylabel('HICP rate')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

#Random Forest forecasting HICP rates EU wide
#using rolling one step forecast as well

x_eu = lag_datasets['EU'].drop(columns=['Target_HICP'])
y_eu = lag_datasets['EU']['Target_HICP']

split_rf = int(len(x_eu) * 0.8)
x_train = x_eu.iloc[:split_rf]
x_test = x_eu.iloc[split_rf:]
y_train = y_eu.iloc[:split_rf]
y_test = y_eu.iloc[split_rf:]

print(f" Train period: {len(x_train)} months")
print(f" Test peirod: {len(x_test)} months")

#Rolling one step forecast
print("\n Running rolling one step forecast")
history_X = x_train.copy()
history_y = y_train.copy()
rf_predictions = []
oob_scores = []

for t in range(len(x_test)):
    rf_forecast = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=42)
    rf_forecast.fit(history_X, history_y)
    oob_scores.append(rf_forecast.oob_score_)

    current_features = x_test.iloc[[t]]
    current_target = y_test.iloc[[t]]
    pred = rf_forecast.predict(current_features)[0]
    rf_predictions.append(pred)

    history_X = pd.concat([history_X, current_features])
    history_y = pd.concat([history_y, y_test.iloc[[t]]])

print(f"  Average OOB R² Score across all instances: {np.mean(oob_scores):.4f}")

#Transforming RF results from differenced to original scale
#For comparison with real HICP rates and meaningful MAE and RMSE interpretation
#Each prediction is added to the real value of the previous month
hicp_eu_original = hicp_rate_wide.mean(axis=1)
rf_predictions_original = []

for idx in range(len(x_test)):
    prev_date = x_test.index[idx] - pd.DateOffset(months=1)
    real_prev_value = hicp_eu_original.loc[prev_date]
    rf_predictions_original.append(real_prev_value + rf_predictions[idx])

rf_predictions_original = pd.Series(rf_predictions_original, index=x_test.index)

#Alligning the real values with corresponding dates
test_original_rf = hicp_eu_original.loc[x_test.index]
train_original_rf = hicp_eu_original.loc[x_train.index]

#Evaluation on the original scale
rf_mae = mean_absolute_error(test_original_rf, rf_predictions_original)
rf_rmse = np.sqrt(mean_squared_error(test_original_rf, rf_predictions_original))

print(f"\n Random Forest Results")
print(f"  MAE:  {rf_mae:.4f}")
print(f"  RMSE: {rf_rmse:.4f}")

#Figure 9 - RF rolling one step forecast compared to actual HICP rate
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_original_rf.index, train_original_rf, color='black', label='Train', linewidth=1)
ax.plot(test_original_rf.index, test_original_rf, color='blue', label='Actual', linewidth=1.5)
ax.plot(rf_predictions_original.index, rf_predictions_original, color='red', label='RF Forecast', linewidth=1.5, linestyle='--')
ax.set_title('Random Forest rolling one step forecast & actual HICP rate')
ax.set_xlabel('Date')
ax.set_ylabel('Annual Rate of Change (%)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_forecast.png', dpi=150, bbox_inches='tight')
plt.show()



#LSTM rolling fine tuning forecasting HICP rates for the EU
#with MinMaxScaling and 6 month look back window

print("\n LSTM forecasting")

#EU wide lagged dataset, split 80/20
x_eu = lag_datasets['EU'].drop(columns=['Target_HICP'])
y_eu = lag_datasets['EU'][['Target_HICP']]

split_lstm = int(len(x_eu) * 0.8)
x_train = x_eu.iloc[:split_lstm]
x_test = x_eu.iloc[split_lstm:]
y_train = y_eu.iloc[:split_lstm]
y_test = y_eu.iloc[split_lstm:]

print(f" Train period: {len(x_train)} months")
print(f" Test period: {len(x_test)} months")

#Scaling features individually based on those used in the train period
#test features are only transformed based on the train period
x_train_scaled = pd.DataFrame(index=x_train.index)
x_test_scaled = pd.DataFrame(index=x_test.index)

for col in x_eu.columns:
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled[col] = scaler.fit_transform(x_train[[col]]).flatten()
    x_test_scaled[col] = scaler.transform(x_test[[col]]).flatten()

#Scaling target as well, same as above
target_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = target_scaler.fit_transform(y_train).flatten()
y_test_scaled = target_scaler.transform(y_test).flatten()

#Building sequences using a 6 month look back window
#with all 24 features to predict the next month

look_back = 6
def sequences(x_data, y_data, look_back):
    x_seq, y_seq = [], []
    for i in range(look_back, len(x_data)):
        x_seq.append(x_data[i - look_back:i])
        y_seq.append(y_data[i])
    return np.array(x_seq), np.array(y_seq)

#Combining train and test so the sequences after the split can still use the lookback window

all_x = pd.concat([x_train_scaled, x_test_scaled])
all_y = np.concatenate([y_train_scaled, y_test_scaled])
all_x_seq, all_y_seq = sequences(all_x.values, all_y, look_back)

#Remaking the split after building the sequences since
#the first 6 rows are lost
split_seq = split_lstm - look_back

x_train_seq = all_x_seq[:split_seq]
y_train_seq = all_y_seq[:split_seq]
x_test_seq = all_x_seq[split_seq:]
y_test_seq = all_y_seq[split_seq:]

print(f" Train sequences: {x_train_seq.shape}")
print(f" Test sequences: {x_test_seq.shape}")

n_features = x_train_seq.shape[2]

#LSTM - 50 neurons, using dense for a single value output

model = Sequential([
    LSTM(50, input_shape=(look_back, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

#Training the model - 50 epochs on the whole training set
print("\n Training LSTM")
model.fit(x_train_seq, y_train_seq, epochs=50, batch_size=16, verbose=0)

#Rolling fine tuning forecast, comparable with walk forward like with RF and ARIMA
#after the initial prediction, the model is fine tuned again
#for a coupl of epochs using the real values, before predicting the next month

print("\n Running LSTM fine tuning forecast")
lstm_predictions_scaled = []

for t in range(len(x_test_seq)):
    current_seq = x_test_seq[t].reshape(1, look_back, n_features)
    pred_scaled = model.predict(current_seq, verbose=0)[0][0]
    lstm_predictions_scaled.append(pred_scaled)

    #fine tuning
    real_value = y_test_seq[t]
    model.fit(current_seq, np.array([real_value]), epochs=5, batch_size=1, verbose=0)

#Inverse transforming, prediction is still differenced
lstm_predictions_diff = target_scaler.inverse_transform(
    np.array(lstm_predictions_scaled).reshape(-1, 1)
).flatten()

#Alligning predictions with the corresponding test dates
test_dates = x_test.index

hicp_eu_original = hicp_rate_wide.mean(axis=1)
lstm_predictions_original = []

for idx in range(len(test_dates)):
    prev_date = test_dates[idx] - pd.DateOffset(months=1)
    real_prev_value = hicp_eu_original.loc[prev_date]
    lstm_predictions_original.append(real_prev_value + lstm_predictions_diff[idx])

lstm_predictions_original = pd.Series(lstm_predictions_original, index=test_dates)

#Original valus, alligned to the same time periods
test_original_lstm = hicp_eu_original.loc[test_dates]
train_original_lstm = hicp_eu_original.loc[x_train.index]

#Evaluation on original scale
lstm_mae = mean_absolute_error(test_original_lstm, lstm_predictions_original)
lstm_rmse = np.sqrt(mean_squared_error(test_original_lstm, lstm_predictions_original))

print(f"\n LSTM Results:")
print(f" MAE: {lstm_mae:.4f}")
print(f" RMSE: {lstm_rmse:.4f}")

#Figure 10 - LSTM rolling fine tuning forecast plotted against to actual HICP rate

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_original_lstm.index, train_original_lstm, color='black', label='Train', linewidth=1)
ax.plot(test_original_lstm.index, test_original_lstm, color='blue', label='Actual', linewidth=1.5)
ax.plot(lstm_predictions_original.index, lstm_predictions_original, color='red', label='LSTM forecast', linewidth=1.5, linestyle='--')
ax.set_title('LSTM rolling fine tuning forecast & actual HICP rate')
ax.set_xlabel('Date')
ax.set_ylabel('Annual Rate of Change (%)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
