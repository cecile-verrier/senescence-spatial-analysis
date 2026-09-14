

import squidpy as sq

from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda import FDataGrid
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from skimage.io import imread
from tqdm import tqdm
from matplotlib.lines import Line2D


import senepy as sp
hubs = sp.load_hubs(species = 'Mouse')

import itertools
import math
import pysal.lib
from pysal.explore import esda
from scipy import stats

import os

print(os.listdir('C:/Users/cecil/Donnee These Stat/data/spatial/heart_mouse_GSE176092/'))

#Data loading and display settings
sc.set_figure_params(dpi_save=300)
plt.rcParams['axes.grid'] = False
files = os.listdir('C:/Users/cecil/Donnee These Stat/data/spatial/heart_mouse_GSE176092/') 

#Function for loading and preprocessing one sample. It retrieves the image and spatial position files,
#loads the expression matrix, filters low-quality spots, normalizes the data, and computes variable genes, 
#PCA, clusters and UMAP projection.

def get_dem_daters(gsm):
    tif = [y for y in [x for x in files if gsm in x] if 'tif' in y][0]
    
    if tif not in files:
        tif = tif.replace('.tif', '_fixed.tif')
    
    
    csv = [y for y in [x for x in files if gsm in x] if 'csv.gz' in y][0]
    pref = csv.split('tissue_positions')[0]
    
    adata = sc.read_10x_mtx('C:/Users/cecil/Donnee These Stat/data/spatial/heart_mouse_GSE176092/', prefix = pref)
    
    df = pd.read_csv('C:/Users/cecil/Donnee These Stat/data/spatial/heart_mouse_GSE176092/' + pref + 'tissue_positions_list.csv.gz',
            header=None, index_col=0)
    
    df = df.loc[adata.obs.index]
    
    adata.obsm['spatial'] = df[[5,4]].values
    adata.obs['array_row'] = df[2]
    adata.obs['array_col'] = df[3]
    img = imread('C:/Users/cecil/Donnee These Stat/data/spatial/heart_mouse_GSE176092/' + tif)
    adata.uns['spatial'] = {}
    adata.uns['spatial']['the_stuff'] = {}
    adata.uns['spatial']['the_stuff']['images'] = {}
    adata.uns['spatial']['the_stuff']['images']['hires'] = img
    
    
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    
    adata = adata[adata.obs.n_genes_by_counts > 1000]
    
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution = 0.3)
    
    return adata


adatas = {}
for file in [x.split('_tissue')[0] for x in files if 'position' in x]:
    gsm = file.split('_')[0]
    adatas[file] = get_dem_daters(gsm)


#Removal of spots outside the tissue for three samples, by filtering on column position.
adatas['GSM5943195_WT_MI_day14_3'] =\
adatas['GSM5943195_WT_MI_day14_3'][adatas['GSM5943195_WT_MI_day14_3'].obs.array_col < 106]

adatas['GSM5943190_WT_MI_day1_2'] =\
adatas['GSM5943190_WT_MI_day1_2'][adatas['GSM5943190_WT_MI_day1_2'].obs.array_col > 20]

adatas['GSM5943192_WT_MI_day7_2'] =\
adatas['GSM5943192_WT_MI_day7_2'][adatas['GSM5943192_WT_MI_day7_2'].obs.array_col < 100]


heart_hubs = {k:v for (k,v) in hubs.hubs.items() if 'Heart' in k[0]}

#Senescence score computation. For each hub, an expression score is calculated per spot, then outlier spots
#are identified (score > mean + 3 standard deviations). The global sen_burden score sums the number of 
#hubs for which a spot is an outlier, and sen_like indicates whether the spot is considered senescent 
#(burden > 0).

def score_the_cells(adata):
    trans = sp.translator(data = adata, hub = heart_hubs)
    for hub in heart_hubs:
        namer = '__'.join([str(x) for x in hub])
        adata.obs[namer] = sp.score_hub(adata, hub = heart_hubs[hub],
                                     translator= trans,
                                        binarize = False, importance = False) #binary false becuase visium higher depth, c'est le score d'expression d'un hub
            
        std = adata.obs[namer].std()
        mu = adata.obs[namer].mean()
        
        adata.obs[namer + '_out'] = (adata.obs[namer] > mu + std*3)*1 #calcul moyenne et ecart type pour chaque spot puis défini un seuil pour "outliers" et creer la nouvelle colonne binaire (0 si pas sene et 1 si sene)
        
    adata.obs['sen_burden'] = adata.obs[[x for x in adata.obs.columns if '_out' in x]].sum(axis = 1)
    adata.obs['sen_like'] = (adata.obs['sen_burden'] > 0)*1
    adata.obs['sen_like'] = adata.obs['sen_like'].astype('category')


for adata in list(adatas):
    score_the_cells(adatas[adata])


#Outlier threshold recalculation using all samples combined. A spot is considered an outlier only if 
#it exceeds both the global threshold (across all samples) and its individual sample threshold, avoiding 
#false positives due to inter-sample variability.

for score in [col.replace('_out', '') for col in adatas[list(adatas)[0]].obs.columns if col.endswith('_out')]:

    combined_scores = pd.concat([df.obs[score] for df in adatas.values()])

    combined_thresh = combined_scores.mean() + 3*combined_scores.std()

    for adata in adatas:
        #above combined thresh and individual sample thresh
        indv_thresh = adatas[adata].obs[score].mean() + 3*adatas[adata].obs[score].std()
        
        adatas[adata].obs[score + '_out'] =\
        ((adatas[adata].obs[score] > combined_thresh) & (adatas[adata].obs[score] > indv_thresh))*1
        
        adatas[adata].obs[score + '_out'] = adatas[adata].obs[score + '_out'].astype('category')
        
        adatas[adata].obs['sen_burden'] =\
        adatas[adata].obs[[x for x in adatas[adata].obs.columns if '_out' in x]].astype(int).sum(axis = 1)


hubs.metadata[hubs.metadata.tissue == 'Heart_and_Aorta']

#Extraction of metadata for each sample: day post-infarction, experimental condition (WT, shCsrp3, Csrp3OE), 
#and proportion of senescent spots.

out = []
for adata in adatas:
    temp = adatas[adata]
    
    if 'day' in adata:
        day = adata.split('day')[1].split('_')[0]
        day = int(day)
    else:
        day = 0
    
    burdened_spots = len(temp.obs[temp.obs.sen_like.astype(int) > 0])
    
    if '_WT_' in adata:
        cond = 'WT'
    elif 'shCsrp3' in adata:
        cond = 'shCsrp3'
    else:
        cond = 'Csrp3OE'
    
    out.append([adata, day, cond, burdened_spots, burdened_spots/len(temp)])
    




df = pd.DataFrame(out, columns = ['id', 'day', 'condition', 'sen', 'per'])
df


df['Rep'] = df['id'].map(lambda x: x.split('_')[-1])
df = df.sort_values('id')
df


import os
os.makedirs('figures', exist_ok=True) 


#Splitting data by experimental group (Sham, day 1, day 7, day 14).
sham = df.loc[df['day'] == 0]['per']
day1 = df.loc[df['day'] == 1]['per']
day7 = df.loc[df['day'] == 7]['per']
day14 = df.loc[df['day'] == 14]['per']


#FIG 2 a) b)
#Visualisation des échantillons day1_1 et day1_3. On affiche d'abord les deux images H&E réelles 
#côte à côte, puis les cartes de score de sénescence (sen_burden) correspondantes avec colorbar.

samples_to_plot = [x for x in adatas if 'day1_1' in x or 'day1_3' in x]

fig, axes = plt.subplots(1, 2, figsize=(7.5, 5))
for ax, sample in zip(axes, samples_to_plot):
    img = adatas[sample].uns['spatial']['the_stuff']['images']['hires']
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()


hub = "Heart_and_Aorta__myocyte__1"

fig, axes = plt.subplots(1, 2, figsize=(8.5, 5))
for ax, sample in zip(axes, samples_to_plot):
    coords = adatas[sample].obsm['spatial']
    score = adatas[sample].obs[hub].values
    sc_plot = ax.scatter(coords[:, 0], coords[:, 1], c=score, cmap='plasma', s=10,
                         vmin=-0.24, vmax=0.03)
    ax.set_title('Sample ' + sample.split('_WT_')[1].replace('MI_day1_', 'Day 1 - Rep. ').replace('MI_day7_', 'Day 7 - Rep. ').replace('MI_day14_', 'Day 14 - Rep. '), fontsize=22)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
cbar = plt.colorbar(sc_plot, ax=axes[1], fraction=0.046, pad=0.04)
cbar.set_label('Score', fontsize=27)
cbar.ax.tick_params(labelsize=27)
plt.tight_layout()
plt.subplots_adjust(wspace=0.5)
plt.show()



## Senescence score visualization (cardiac myocyte hub) for WT samples
hub = "Heart_and_Aorta__myocyte__1"

wt_samples = [
    'GSM5355663_WT_Sham',
    'GSM5943189_WT_MI_day1_1',
    'GSM5943190_WT_MI_day1_2',
    'GSM5943191_WT_MI_day1_3',
    'GSM5355666_WT_MI_day7_1',
    'GSM5943192_WT_MI_day7_2',
    'GSM5943193_WT_MI_day7_3',
    'GSM5355668_WT_MI_day14_1',
    'GSM5943194_WT_MI_day14_2',
    'GSM5943195_WT_MI_day14_3']

groups = [wt_samples[:5], wt_samples[5:]]

for group in groups:

    fig, axes = plt.subplots(2, 5, figsize=(10, 5.7), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08, "wspace": 0.08})

    for col, sample in enumerate(group):

        adata = adatas[sample]

        library_id = list(adata.uns["spatial"].keys())[0]
        spatial_data = adata.uns["spatial"][library_id]

        if "hires" in spatial_data["images"]:
            tissue_image = spatial_data["images"]["hires"]
        elif "lowres" in spatial_data["images"]:
            tissue_image = spatial_data["images"]["lowres"]
        else:
            raise KeyError(f"Aucune image hires ou lowres trouvée pour {sample}")

        spatial_coordinates = adata.obsm["spatial"].astype(float)

        nom = ("Sham" if "Sham" in sample else sample.split("_WT_MI_")[1].replace("day1_", "Day 1 - Rep. ").replace("day7_", "Day 7 - Rep. ").replace("day14_", "Day 14 - Rep. "))

        # Image histologique réelle
        ax_image = axes[0, col]
        ax_image.imshow(tissue_image)
        ax_image.set_title(nom, fontsize=33)
        ax_image.set_aspect("equal")
        ax_image.axis("off")

        # Carte du score
        ax_score = axes[1, col]
        sc = ax_score.scatter(spatial_coordinates[:, 0], spatial_coordinates[:, 1], c=adata.obs[hub], cmap="plasma", s=6, vmin=-0.24, vmax=0.03)

        ax_score.set_aspect("equal")
        ax_score.invert_yaxis()
        ax_score.axis("off")
        ax_score.set_adjustable("datalim")

    cbar = fig.colorbar(sc, ax=axes[1, :], orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("Score", fontsize=30)
    cbar.ax.tick_params(labelsize=30)

    fig.subplots_adjust(left=0.01, right=0.92, top=0.90, bottom=0.04)

    plt.show()
    

#FIG 2 c) d)
#For each WT sample, computation of the observed L(r) function on senescent spots, then permutation test
#(B=199) to build the CSR reference curve and associated p-values. Results are stored distance by distance 
#in out_2.

B = 199
rng = np.random.default_rng(0)
out_2 = []
  

for adata in adatas:
    
    ad = adatas[adata] 

    if '_WT_' not in adata: #only WT sample
        continue

    if 'day' in adata:
        day = int(adata.split('day')[1].split('_')[0])
    else:
        day = 0

    if '_WT_' in adata:
        cond = 'WT'
    elif 'shCsrp3' in adata:
        cond = 'shCsrp3'
    else:
        cond = 'Csrp3OE'

    num_sen_like = (ad.obs['sen_like'] == 1).sum()    
    ripley_result = sq.gr.ripley(ad, cluster_key='sen_like', mode="L", max_dist=250, n_steps=100, copy=True) 
    
    ripley_df = ripley_result['L_stat']
    mask = (ripley_df['sen_like'] == 1).values
    ripley_df = ripley_df.loc[mask].reset_index(drop=True)
    
    num_tot = ad.shape[0]
    
    bins_obs = ripley_df['bins'].values 
    L_obs = ripley_df['stats'].values 
    
    perm_mat = np.empty((B, len(bins_obs))) 
    base = ad.obs['sen_like'] 
    
    for b in range(B): #Permutation test (B=199 times) to build the CSR reference curve.
        perm = rng.permutation(base) 
        ad.obs['sen_like_perm'] = pd.Categorical(perm) 
    
        res_perm = sq.gr.ripley(ad, cluster_key='sen_like_perm', mode="L", max_dist=250, n_steps=100, copy=True) 
        dfp = res_perm['L_stat'] 
        dfp1 = dfp[dfp["sen_like_perm"] == 1]  
        perm_mat[b, :] = dfp1["stats"].to_numpy()  
    
    del ad.obs['sen_like_perm']
    
    sim_mean = perm_mat.mean(axis=0) 
    sim_q025 = np.quantile(perm_mat, 0.025, axis=0)   #Lower 95% confidence bound
    sim_q975 = np.quantile(perm_mat, 0.975, axis=0)   #Upper 95% confidence bound
    
    pvalues = (1 + (perm_mat >= L_obs[None, :]).sum(axis=0)) / (B + 1) 

    try:
        i = np.where(ad.var_names == 'Col1a1')[0][0]
        a = ad.X[:, i].toarray().flatten()
        pear = stats.pearsonr(a, ad.obs['sen_like'].astype(int))
    except:
        pear = (np.nan, np.nan) 

    for j, r_value in enumerate(bins_obs):
        L_stat = L_obs[j]
        p_val = pvalues[j] 
        out_2.append([cond, day, adata, r_value, L_stat, p_val, num_sen_like, num_tot, pear[0], pear[1], sim_mean[j], sim_q025[j], sim_q975[j]])


#final DataFrame with L(r) and permutation results.
columns_2 = ['cond', 'day', 'adata', 'r_value', 'L_stat', 'p_val', 'num_sen_like', 'num_tot', 'pear_corr', 'pear_pval', 'sim_mean', 'sim_q025', 'sim_q975']
df_out_2 = pd.DataFrame(out_2, columns=columns_2)



    
import itertools
import math
import pysal.lib
from pysal.explore import esda

#Function to compute Moran's I. It builds the neighbourhood matrix between spots (distance below the threshold), 
#weighted by the inverse distance, then computes spatial autocorrelation via PySAL.
#Cette fonction est reprise de l'article de Sanborn et al.

def spatial_ac(adata, obs_value, ethresh=3):
    spots = np.vstack((adata.obs.index, adata.obs.array_row.values, adata.obs.array_col.values)).T.tolist()
    pairs = list(itertools.combinations(spots, 2))

    def get_edis(spot1, spot2):
        return math.sqrt((spot1[0] - spot2[0])**2 + (spot1[1] - spot2[1])**2)

    out = []
    for pair in pairs:
        spot1, spot2 = pair
        out.append([spot1[0], spot2[0], get_edis((spot1[1], spot1[2]), (spot2[1], spot2[2]))])

    dists = np.array([x for x in out if x[2] <= ethresh])

    bcs = np.unique(dists[:, 0:2])

    neighbors = {}
    weights = {}
    for bc in bcs:
        temp_n = []
        temp_w = []
        temp = dists[(dists[:, 0] == bc) | (dists[:, 1] == bc)]
        for row in temp.tolist():
            a, b, c = row
            if a == bc:
                temp_n.append(b)
            else:
                temp_n.append(a)
            temp_w.append(1 / float(c))
        neighbors[bc] = temp_n
        weights[bc] = temp_w

    w = pysal.lib.weights.W(neighbors, weights=weights) # builds the spatial weight matrix
    y_vals = dict(zip(adata.obs.index, adata.obs[obs_value]))
    y = np.vectorize(y_vals.get)(bcs)                   # retrieves values in neighbour order

    return esda.Moran(y, w, two_tailed=False)


out_moran = []

for adata in adatas:
    if '_WT_' not in adata:
        continue

    ad = adatas[adata]

    # Moran's I via PySAL (permutations internes)
    res = spatial_ac(ad, 'sen_like', ethresh=2)

    # Pearson for Col1a1
    try:
        i = np.where(ad.var_names == 'Col1a1')[0][0]
        a = ad.X[:, i].toarray().flatten()
        pear = stats.pearsonr(a, ad.obs['sen_like'].astype(int))
        pear_stat, pear_pval = pear.statistic, pear.pvalue
    except Exception:
        pear_stat, pear_pval = np.nan, np.nan

    
    if 'day' in adata:
        day = int(adata.split('day')[1].split('_')[0])
    else:
        day = 0

    out_moran.append([adata, day, 'WT',res.I,res.p_z_sim,res.p_sim,res.EI,pear_stat,pear_pval])

sample_stats_wt = pd.DataFrame(
    out_moran,
    columns=['ID', 'day', 'cond', 'I', 'p_z_sim', 'p_sim', 'EI', 'col_corr_stat', 'col_corr_pval'])

sample_stats_wt['Rep'] = sample_stats_wt['ID'].map(lambda x: x.split('_')[-1])
sample_stats_wt = sample_stats_wt.sort_values('ID')



#Number of senescent spot
for adata_name, adata in adatas.items():
    nb_sen = (adata.obs['sen_like'].astype(int) == 1).sum()
    nb_tot = adata.n_obs
    print(f"{adata_name} : {nb_sen} / {nb_tot} senescents spots")

# GSM5355663_WT_Sham : 125 / 1684 spots sénescents
# GSM5355666_WT_MI_day7_1 : 163 / 3151 spots sénescents
# GSM5355668_WT_MI_day14_1 : 28 / 1152 spots sénescents
# GSM5943189_WT_MI_day1_1 : 67 / 1570 spots sénescents
# GSM5943190_WT_MI_day1_2 : 89 / 1553 spots sénescents
# GSM5943191_WT_MI_day1_3 : 78 / 1452 spots sénescents
# GSM5943192_WT_MI_day7_2 : 60 / 913 spots sénescents
# GSM5943193_WT_MI_day7_3 : 89 / 1215 spots sénescents
# GSM5943194_WT_MI_day14_2 : 118 / 1796 spots sénescents
# GSM5943195_WT_MI_day14_3 : 68 / 1390 spots sénescents
# GSM5943196_shCsrp3_MI_day1 : 66 / 1410 spots sénescents
# GSM5943197_shCsrp3_MI_day14 : 99 / 1520 spots sénescents
# GSM5943198_Csrp3OE_MI_day1 : 34 / 1029 spots sénescents
# GSM5943199_Csrp3OE_MI_day14 : 93 / 1520 spots sénescents

colors = {'WT': 'black', 'Csrp3OE': 'blue', 'shCsrp3': 'red'}

#mean and median L(r) curve across all WT samples, with the CSR reference curve from permutations.
plt.figure(figsize=(10,8)) 
r_values = sorted(df_out_2['r_value'].unique())

for cond in ['WT']: 
    
    df_ref = df_out_2[df_out_2['cond'] == cond].groupby('r_value')['sim_mean'].mean()
    plt.plot(df_ref.index, df_ref.values, label='Reference CSR', color=colors[cond], linestyle=':')
 
    df_cond_2 = df_out_2[df_out_2['cond'] == cond].groupby('r_value')['L_stat'] 
    summary_L_mean_2 = df_cond_2.mean()
    summary_L_median_2 = df_cond_2.median()
    print(f"{cond} - Median r-values:", summary_L_median_2.index.tolist())
   
    #mean
    plt.plot(summary_L_mean_2.index, summary_L_mean_2.values, label='Mean', color=colors[cond], linestyle='-')
    #mediane
    plt.plot(summary_L_median_2.index, summary_L_median_2.values, label= 'Mediane', color=colors[cond], linestyle='--')


plt.xlabel("Distance r", fontsize=28)
plt.ylabel("L(r)", fontsize=28)
plt.title("")
plt.legend(fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

#Visualisation of the L(r) curve for each sample individually
df_wt = df_out_2[df_out_2['cond'] == 'WT'].copy()

df_wt['label'] = 'Sample ' + df_wt['adata'].str.split('_WT_').str[1]
df_wt['label'] = df_wt['label'].str.replace('Sham', 'Sham')
df_wt['label'] = df_wt['label'].str.replace('MI_day1_', 'Day 1 - Rep. ')
df_wt['label'] = df_wt['label'].str.replace('MI_day7_', 'Day 7 - Rep. ')
df_wt['label'] = df_wt['label'].str.replace('MI_day14_', 'Day 14 - Rep. ')

plt.figure(figsize=(16,8))
sns.lineplot(data=df_wt, x='r_value', y='L_stat', hue='label', err_style=None)
df_ref = df_out_2[df_out_2['cond'] == 'WT'].groupby('r_value')['sim_mean'].mean()
plt.plot(df_ref.index, df_ref.values, 'k--', label='CSR reference')
plt.xlabel('Distance r', fontsize=31)
plt.ylabel('L(r)', fontsize=31)
plt.subplots_adjust(left=0.41)
plt.legend(fontsize=28, loc='upper left', bbox_to_anchor=(-0.70, 1))
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

#FIG 3
#Bootstrap on WT L(r) curves and visualisation of the mean, median, 95% confidence interval and CSR reference curve.

courbes = []
adata_list = []

for adata_name in df_out_2["adata"].unique():
    
    df_tmp = df_out_2[df_out_2["adata"] == adata_name].sort_values("r_value")
    r = df_tmp["r_value"].values      
    y = df_tmp["L_stat"].values       

    courbes.append(y)
    adata_list.append(adata_name) 

R = np.array(courbes)                   
adata_list = np.array(adata_list) 
n = 10 
B = 1000

R_boot_all = []

for b in range(B):
    tirage = np.random.choice(n, size=n, replace=True)
    R_b = R[tirage] 
    R_boot_all.append(R_b)

R_boot_all = np.array(R_boot_all)

cond_per_sample = []

for adata_name in adata_list:
    cond = df_out_2[df_out_2["adata"] == adata_name]["cond"].iloc[0]
    cond_per_sample.append(cond)

cond_per_sample = np.array(cond_per_sample) 

cond_WT = cond_per_sample == "WT"
ref_WT = (df_out_2[df_out_2["cond"] == "WT"].sort_values("r_value").groupby("r_value", as_index=False)["sim_mean"].mean())["sim_mean"].to_numpy()
R_boot_WT  = R_boot_all[:, cond_WT, :]


WT_boot_mean   = np.mean(R_boot_WT, axis=1) # mean and median across bootstrap samples
WT_boot_median = np.median(R_boot_WT, axis=1)
WT_mean   = np.mean(WT_boot_mean, axis=0)
WT_median = np.median(WT_boot_median, axis=0)
WT_low_mean  = np.percentile(WT_boot_mean, 2.5, axis=0) # 95% confidence interval bounds
WT_high_mean = np.percentile(WT_boot_mean, 97.5, axis=0) 


plt.figure(figsize=(13,8))
plt.plot(r, WT_mean, color="black", label="Mean")
plt.fill_between(r, WT_low_mean, WT_high_mean, color="black", alpha=0.25, label="95% CI")
plt.plot(r, WT_median, color="black", linestyle="--", label="Median")
plt.plot(r, ref_WT, color="black", linestyle=":", label="CSR reference")
plt.xlabel("Distance r", fontsize=28)
plt.ylabel("L(r)", fontsize=28)
plt.title("Bootstrap", fontsize=28)
plt.legend(fontsize=28, loc='upper left')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()


def boot_stats(mat):
    return mat.mean(axis=0), np.median(mat, axis=0), np.percentile(mat, 2.5, axis=0), np.percentile(mat, 97.5, axis=0)

from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
import numpy as np
import matplotlib.pyplot as plt
from skfda.ml.clustering import KMeans as FKMeans
import matplotlib.patches as mpatches

r_vals = sorted(df_out_2[df_out_2['r_value'] >= 40]['r_value'].unique())

#FIG 4
#FDA pipeline on L(r) curves for WT samples. Curves are recentred relative to the CSR reference, 
#smoothed using B-splines, then submitted to FPCA (2 components). A bootstrap (B=100) validates the stability
#of the mean curve and first principal component.

for cond in ['WT']:
    graph_cond = df_out_2[df_out_2['cond'] == cond].copy()

    courbes = []
    adata_names = graph_cond['adata'].unique()

    for adata_name in adata_names:
        df_sample = graph_cond[graph_cond['adata'] == adata_name].copy()
        df_sample = df_sample[df_sample['r_value'] >= 40]

        L_values = (df_sample['L_stat'] - df_sample['sim_mean']).values  #Centre each L(r) curve relative to its permutation-based CSR reference
        #L_values = df_sample['L_stat'].values
        courbes.append(np.asarray(L_values).flatten())

    if len(courbes) < 2:
        continue

    fd = FDataGrid(data_matrix=courbes, grid_points=r_vals)

    basis = BSplineBasis(domain_range=(np.min(r_vals), np.max(r_vals)), n_basis=20) # smooth curves onto a B-spline basis (20 basis functions) to reduce noise before FPCA
    smoother = BasisSmoother(basis=basis)
    fd_smooth = smoother.fit_transform(fd)

    mean_curve = fd_smooth.mean()

    plt.figure(figsize=(10, 6))
    for f in fd_smooth:
        plt.plot(r_vals, f(r_vals)[0], color='lightgray', alpha=0.7)
    plt.plot(r_vals, mean_curve(r_vals)[0], color='blue', linewidth=2.5, label='Functional mean')
    plt.title("")
    plt.xlabel("Distance r", fontsize=28)
    plt.ylabel("L(r)", fontsize=28)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.legend(fontsize=28)
    plt.tight_layout()
    plt.show()

    fpca = FPCA(n_components=2)
    fpca.fit(fd_smooth)
    scores = fpca.transform(fd_smooth)

    print(f"Condition: {cond}")
    print(f"Number of curves: {len(courbes)}")
    print("FPCA scores (first 2 components):")
    print(scores)
    print("Variance explained by each component:", fpca.explained_variance_ratio_)

    plt.figure(figsize=(10, 8))
    for i, component in enumerate(fpca.components_):
        plt.plot(r_vals, component(r_vals)[0], label=f'PC{i+1}')
    plt.title("")
    plt.xlabel("Distance r", fontsize=31)
    plt.ylabel("Component value", fontsize=31)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=28)
    plt.tight_layout()
    plt.show()
    
    # === Bootstrap FPCA ===
    B = 100
    R = np.array(courbes)
    n = R.shape[0]

    pc1_ref = fpca.components_[0](r_vals)[0].flatten()

    R_boot_all = []
    for b in range(B):
        tirage = np.random.choice(n, size=n, replace=True) #generate a bootstrap sample of curves by resampling with replacement
        R_boot_all.append(R[tirage])
    R_boot_all = np.array(R_boot_all)

    pc1_boot = []
    mean_boot = []
    var_exp_boot = []

    for b in range(B):
        R_b = R_boot_all[b]

        fd_b = FDataGrid(data_matrix=R_b, grid_points=r_vals)
        fd_b_smooth = BasisSmoother(basis).fit_transform(fd_b)
        fpca_b = FPCA(n_components=2)
        fpca_b.fit(fd_b_smooth)

        phi1 = fpca_b.components_[0](r_vals)[0].flatten()

        if np.dot(phi1, pc1_ref) < 0:
            phi1 = -phi1 #Align the orientation of bootstrap eigenfunctions with the reference PC1

        pc1_boot.append(phi1)
        mean_boot.append(fd_b_smooth.mean()(r_vals)[0].flatten())
        var_exp_boot.append(fpca_b.explained_variance_ratio_[0])

    pc1_boot = np.array(pc1_boot)
    mean_boot = np.array(mean_boot)
    var_exp_boot = np.array(var_exp_boot)

    pc1_mean, pc1_med, pc1_low, pc1_high = boot_stats(pc1_boot)
    mean_mean, mean_med, mean_low, mean_high = boot_stats(mean_boot)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_vals, mean_mean, color='black', label='Mean')
    ax.plot(r_vals, mean_med, color='black', linestyle='--', label='Median')
    ax.fill_between(r_vals, mean_low, mean_high, alpha=0.25, color='black', label='95% CI')
    ax.set_xlabel('Distance r', fontsize=23)
    ax.set_ylabel('L(r)', fontsize=23)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #ax.set_title('Bootstrapped mean curve — L(r)')
    ax.legend(fontsize=23)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_vals, pc1_mean, color='black', label='Mean')
    ax.plot(r_vals, pc1_med, color='black', linestyle='--', label='Median')
    ax.fill_between(r_vals, pc1_low, pc1_high, alpha=0.25, color='black', label='95% CI')
    ax.set_xlabel('Distance r', fontsize=23)
    ax.set_ylabel('PC1 - L(r)', fontsize=23)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #ax.set_title(f'PC1 — variance explained: {var_exp_boot.mean()*100:.1f}%')
    ax.legend(fontsize=23)
    plt.tight_layout()
    plt.show()
    


from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis, FDataBasis
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.ml.clustering import KMeans as FKMeans


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


courbes = []
meta = []

for adata_name in df_out_2["adata"].unique():

    df_tmp = df_out_2[df_out_2["adata"] == adata_name].sort_values("r_value") 
    df_tmp = df_tmp[df_tmp["r_value"] >= 40]
    if len(df_tmp) < 5: 
        continue

    
    r = df_tmp["r_value"].values 
    #y = df_tmp["L_stat"].values 
    y = (df_tmp["L_stat"] - df_tmp["sim_mean"]).values
    courbes.append((r, y)) 

    cond = df_tmp["cond"].iloc[0] 
    meta.append((cond, adata_name)) 

meta = pd.DataFrame(meta, columns=["cond", "adata"]) 


domain_min = 40
domain_max = 250
basis = BSplineBasis(domain_range=(domain_min, domain_max),n_basis=20) 
courbes_basis = []

for (r, y) in courbes: 
    fd_i = FDataGrid(data_matrix=[y], grid_points=[r]) 
    fd_i_basis = fd_i.to_basis(basis) 
    courbes_basis.append(fd_i_basis) 


coeffs = np.vstack([fd.coefficients[0] for fd in courbes_basis]) 
fd_all = FDataBasis(basis=basis, coefficients=coeffs)
grid = np.linspace(domain_min, domain_max, 200)
evaluated = fd_all(grid)
fd_regular = FDataGrid(evaluated,grid) 

#FPCA
fpca = FPCA(n_components=2) 
scores = fpca.fit_transform(fd_regular)

meta["PC1"] = scores[:, 0]
meta["PC2"] = scores[:, 1]

#FIG 5
#FPCA projection according to time post-infarction

meta["day"] = meta["adata"].apply(lambda x: int(x.split("day")[1].split("_")[0]) if "day" in x else 0)

palette = {0: "grey", 1: "red", 7: "green", 14: "blue"}
labels = {0: "Sham sample", 1: "Day 1 samples", 7: "Day 7 samples", 14: "Day 14 samples"}

plt.figure(figsize=(17, 9))          # 14 + 5 pouces de blanc à gauche

for d in [0, 1, 7, 14]:
    sel = meta["day"] == d
    plt.scatter(meta.loc[sel, "PC1"], meta.loc[sel, "PC2"],
                s=80, color=palette[d], label=labels[d])

plt.xlabel("PC1", fontsize=28)
plt.ylabel("PC2", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.subplots_adjust(left=0.30, right=0.93)
plt.legend(fontsize=30, loc="upper left", bbox_to_anchor=(-0.45, 1))
plt.show()

#Functional clustering
kmeans = FKMeans(n_clusters=2, random_state=42)
meta["cluster"] = kmeans.fit_predict(fd_regular)

plt.figure(figsize=(15, 9))
for cl in meta["cluster"].unique():
    clus = meta["cluster"] == cl
    plt.scatter(meta.loc[clus, "PC1"], meta.loc[clus, "PC2"], s=80, label=f"Cluster {cl}")

plt.xlabel("PC1", fontsize=28)
plt.ylabel("PC2", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(loc="upper left", bbox_to_anchor=(-0.35, 1), fontsize=30)
plt.subplots_adjust(left=0.25)
plt.show()


R = np.array([y for (r_tmp, y) in courbes])
n = R.shape[0]
scores_boot = []  

sim_pc1 = []
sim_pc2 = []
pc1_ref = None
pc2_ref = None

tirages_saved = []

for b in range(B):
    tirage2 = np.random.choice(n, size=n, replace=True)
    tirages_saved.append(tirage2)
    R_b = R[tirage2]

    courbes_basis = []

    for y in R_b:
        fd_i = FDataGrid(data_matrix=[y], grid_points=[r])
        fd_i_basis = fd_i.to_basis(basis)
        courbes_basis.append(fd_i_basis) 

    coeffs = np.vstack([fd.coefficients[0] for fd in courbes_basis])
    fd_all = FDataBasis(basis=basis, coefficients=coeffs)

    evaluated = fd_all(grid)
    fd_regular = FDataGrid(evaluated, grid)

    #FPCA
    fpca = FPCA(n_components=2)
    scores = fpca.fit_transform(fd_regular)
    scores_boot.append(scores) 

    pc1 = fpca.components_[0](grid).ravel()
    pc2 = fpca.components_[1](grid).ravel()
    
    if pc1_ref is None:
        pc1_ref = pc1
        pc2_ref = pc2
    
    def sim(a, b):
        return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    sim_pc1.append(sim(pc1_ref, pc1))
    sim_pc2.append(sim(pc2_ref, pc2))


print("PC1 similarity mean/min:", np.mean(sim_pc1), np.min(sim_pc1))
print("PC2 similarity mean/min:", np.mean(sim_pc2), np.min(sim_pc2))
print("PC2 < 0.9 proportion:", np.mean(np.array(sim_pc2) < 0.9))

# PC1 similarity mean/min: 0.9993530753001568 0.9908398670475707
# PC2 similarity mean/min: 0.8780887132518832 0.2156900652897427
# PC2 < 0.9 proportion: 0.038

#FIG 6a
#Visualisation of FPCA scores across all bootstrap samples, coloured by day post-infarction.

day_per_sample = np.array([df_out_2[df_out_2["adata"] == name]["day"].iloc[0] for name in adata_list])

days_all = np.concatenate([day_per_sample[t] for t in tirages_saved])
scores_boot = np.array(scores_boot)
X_all = scores_boot.reshape(-1, 2)

palette = {0: "grey", 1: "red", 7: "green", 14: "blue"}
labels = {0: "Sham sample", 1: "Day 1 samples", 7: "Day 7 samples", 14: "Day 14 samples"}
colors_all = [palette[d] for d in days_all]

plt.figure(figsize=(16, 9))
plt.scatter(X_all[:, 0], X_all[:, 1], c=colors_all, alpha=0.5, s=20)

legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[d], markersize=8, label=labels[d]) for d in [0, 1, 7, 14]]

plt.xlabel("PC1", fontsize=30)
plt.ylabel("PC2", fontsize=30)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(-0.55, 1.05), fontsize=34)
plt.subplots_adjust(left=0.35)
plt.show()

#FIG 6b
#similarite cosinus
import numpy as np
import matplotlib.pyplot as plt

def cos_sim_abs(a, b):
    return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def pre_fpca(R_mat, r, basis, grid): 
    courbes_basis = []
    for y in R_mat:
        fd_i = FDataGrid(data_matrix=[y], grid_points=[r]) #convert one observed L(r) curve into a functional object
        fd_i_basis = fd_i.to_basis(basis) 
        courbes_basis.append(fd_i_basis) 

    coeffs = np.vstack([fd.coefficients[0] for fd in courbes_basis])
    fd_all = FDataBasis(basis=basis, coefficients=coeffs)

    evaluated = fd_all(grid)
    return FDataGrid(evaluated, grid)

n = R.shape[0]

fd_regular_ref = pre_fpca(R, r, basis, grid)   # smooth all curves on a common grid before FPCA
fpca_ref = FPCA(n_components=2)
scores_ref = fpca_ref.fit_transform(fd_regular_ref) # compute the first two principal components
pc1_ref = fpca_ref.components_[0](grid).ravel()
pc2_ref = fpca_ref.components_[1](grid).ravel()

plt.figure(figsize=(16, 9))
plt.hist(sim_pc2, bins=20, alpha=0.5, color='red', label="PC2", edgecolor='black')
plt.hist(sim_pc1, bins=20, histtype='step', linewidth=4, color='blue', label="PC1")

plt.xlabel("Cosine similarity (absolute value)", fontsize=28)
plt.ylabel("Frequency", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=30)
plt.show()

#FIG 7
#Bootstrap on the mean L(r) curve and visualisation of FPCA score stability

courbes = []
adata_list = []

for adata_name in df_out_2["adata"].unique():
    df_tmp = df_out_2[df_out_2["adata"] == adata_name].sort_values("r_value")

    r = df_tmp["r_value"].values
    #y = df_tmp["L_stat"].values
    y = (df_tmp["L_stat"] - df_tmp["sim_mean"]).values
    courbes.append(y)
    adata_list.append(adata_name)

R = np.array(courbes)
adata_list = np.array(adata_list)

courbes_basis_ref = []
for y in R:
    fd_i = FDataGrid(data_matrix=[y], grid_points=[r])
    fd_i_basis = fd_i.to_basis(basis)
    courbes_basis_ref.append(fd_i_basis)

coeffs_ref = np.vstack([fd.coefficients[0] for fd in courbes_basis_ref])
fd_all_ref = FDataBasis(basis=basis, coefficients=coeffs_ref)

evaluated_ref = fd_all_ref(grid)
fd_regular_ref = FDataGrid(evaluated_ref, grid)

fpca_ref = FPCA(n_components=2)
fpca_ref.fit(fd_regular_ref)

B = 300
n = R.shape[0]
scores_boot = []

for b in range(B):
    tirage2 = np.random.choice(n, size=n, replace=True)
    R_b = R[tirage2]

    y_mean = R_b.mean(axis=0)

    fd_mean = FDataGrid(data_matrix=[y_mean], grid_points=[r])
    fd_mean_basis = fd_mean.to_basis(basis)
    fd_mean_all = FDataBasis(basis=basis, coefficients=fd_mean_basis.coefficients)

    evaluated_mean = fd_mean_all(grid)
    fd_regular_mean = FDataGrid(evaluated_mean, grid)

    score = fpca_ref.transform(fd_regular_mean)[0]
    scores_boot.append(score)

scores_boot = np.array(scores_boot)
scores_ref = fpca_ref.transform(fd_regular_ref)   # (10,2)
mu = scores_ref.mean(axis=0)

plt.figure(figsize=(16, 10))
plt.scatter(scores_boot[:, 0], scores_boot[:, 1], s=20, label="Bootstrap means")
plt.scatter(mu[0], mu[1], s=120, label="Mean of 10 samples")
plt.xlabel("PC1", fontsize=28)
plt.ylabel("PC2", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(loc="upper left", fontsize=30)
plt.show()


#PCF
#FIG 8

import sys
sys.path.append(r"C:\Users\cecil\Thèse  partie stat")
from helperFunctions import generatePointCloud, pairCorrelationFunction 

import numpy as np
import matplotlib.pyplot as plt
from helperFunctions import topographicalCorrelationMap


samples = {
    "day1": "GSM5943189_WT_MI_day1_1",
    "day7": "GSM5943192_WT_MI_day7_2",
    "day14": "GSM5943195_WT_MI_day14_3"}

B = 199
rng = np.random.default_rng(0)

for day, sample_name in samples.items():
    adata = adatas[sample_name]
    points = adata.obsm["spatial"].astype(float)
    domain = np.array([
        [points[:, 0].min(), points[:, 0].max()],
        [points[:, 1].min(), points[:, 1].max()],])
    pc = generatePointCloud("sample", points, domain=domain) # store spot coordinates and associated information in a single object
    sen_like = adata.obs["sen_like"].astype(str).to_numpy()
    pc.addLabels("sen_like", "categorical", sen_like)

    # PCF
    r, g_obs, _ = pairCorrelationFunction(
        pc, labelName="sen_like", categoriesToPlot=["1", "1"],
        maxR=250, annulusStep=2.5, annulusWidth=2.5)
    g_obs = g_obs.flatten()
    perm_mat = np.empty((B, len(r)))
    base = adata.obs["sen_like"].astype(str).to_numpy()
    for b in range(B):
        perm = rng.permutation(base)
        pc_perm = generatePointCloud("sample", points, domain=domain)
        pc_perm.addLabels("sen_like", "categorical", perm)
        _, g_perm, _ = pairCorrelationFunction(
            pc_perm, labelName="sen_like", categoriesToPlot=["1", "1"],
            maxR=250, annulusStep=2.5, annulusWidth=2.5)
        perm_mat[b, :] = g_perm.flatten()
        
    sim_mean = perm_mat.mean(axis=0)
    plt.figure(figsize=(12, 8))
    
    plt.plot(r, g_obs, label="Observed")
    plt.plot(r, sim_mean, linestyle="--", color="gray", label="CSR reference")
    plt.xlabel("Distance r", fontsize=41)
    plt.ylabel("g(r) sen-sen", fontsize=41)
    plt.xticks(fontsize=37)
    plt.yticks(fontsize=37)
    plt.legend(fontsize=35)
    plt.title(f"Pair correlation function - {day.replace('day', 'Day ')}", fontsize=41)
    plt.show()

    # TCM
    maxR = 250
    annulusStep = 2.5
    annulusWidth = 2.5
    r, g, contributions = pairCorrelationFunction(
        pc,
        labelName="sen_like",
        categoriesToPlot=["1", "1"],
        maxR=maxR,
        annulusStep=annulusStep,
        annulusWidth=annulusWidth)
    g = g.flatten()
    tcm = topographicalCorrelationMap(pc, "sen_like", "1", "sen_like", "1", radiusOfInterest=100, maxCorrelationThreshold=5.0, kernelRadius=150, kernelSigma=100, visualiseStages=False)
    
    plt.figure(figsize=(12, 11))
    
    l = int(np.ceil(np.max(np.abs([tcm.min(), tcm.max()]))))
    plt.imshow(tcm, cmap="RdBu_r", vmin=-l, vmax=l, origin="lower")
    cbar = plt.colorbar(label=r"$\Gamma_{\mathrm{sen,sen}}(r=100)$")
    cbar.set_label(r"$\Gamma_{\mathrm{sen,sen}}(r=100)$", fontsize=34)
    ax = plt.gca()
    plt.title("", fontsize=43)
    plt.xticks(np.arange(0, tcm.shape[1] + 1, 300), fontsize=36)
    plt.yticks(fontsize=36)
    plt.title(f"Topographic correlation map - {day.replace('day', 'Day ')}", fontsize=43, pad=22)
    plt.tight_layout()
    ax.grid(False)
    plt.show()


##Fig 9
# Overlay of TCM on histological H&E images

for rad in [100]:
    for sample_id in wt_samples:
        adata = adatas[sample_id]

        library_id = list(adata.uns["spatial"].keys())[0]
        tissue_image = adata.uns["spatial"][library_id]["images"]["hires"]

        points = adata.obsm["spatial"].astype(float)
        domain = np.array([[points[:, 0].min(), points[:, 0].max()],
                           [points[:, 1].min(), points[:, 1].max()]])

        pc = generatePointCloud(sample_id, points, domain=domain)
        pc.addLabels("sen_like", "categorical", adata.obs["sen_like"].astype(str).to_numpy())

        tcm = topographicalCorrelationMap(pc, "sen_like", "1", "sen_like", "1",
                                          radiusOfInterest=rad, maxCorrelationThreshold=5.0,
                                          kernelRadius=150, kernelSigma=100, visualiseStages=False)

        l = int(np.ceil(np.max(np.abs([tcm.min(), tcm.max()]))))

        nom = ("Sham sample" if "Sham" in sample_id
               else "Sample " + sample_id.split("_WT_MI_")[1].replace("day1_", "Day 1 - Rep. ")
                                                            .replace("day7_", "Day 7 - Rep. ")
                                                            .replace("day14_", "Day 14 - Rep. "))

        fig, ax = plt.subplots(figsize=(11, 10))
        ax.imshow(tissue_image)
        im = ax.imshow(tcm, cmap="RdBu_r", vmin=-l, vmax=l, alpha=0.50,
                       extent=[domain[0, 0], domain[0, 1], domain[1, 1], domain[1, 0]])

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(rf"$\Gamma_{{\mathrm{{sen,sen}}}}(r={rad})$", fontsize=34)
        cbar.ax.tick_params(labelsize=30)

        ax.set_title(f"TCM overlay on H&E image, {nom}", fontsize=34.5, x=0.62)

        margin_x = 0.2 * (domain[0, 1] - domain[0, 0])
        margin_y = 0.2 * (domain[1, 1] - domain[1, 0])
        ax.set_xlim(domain[0, 0] - margin_x, domain[0, 1] + margin_x)
        ax.set_ylim(domain[1, 1] + margin_y, domain[1, 0] - margin_y)

        ax.axis("off")
        plt.tight_layout()
        plt.show()
        

#FIG 10
#For each WT sample, the PCF of senescent spots is computed, then FPCA is applied to extract and visualise 
#the two main sources of variability across samples.

import sys
sys.path.append(r"C:\Users\cecil\Thèse  partie stat")
from helperFunctions import generatePointCloud, pairCorrelationFunction

pcf_list = []
maxR = 250
annulusStep = 2.5
annulusWidth = 2.5
B_pcf = 199
rng_pcf = np.random.default_rng(0)

for sample_id, adata in adatas.items():
    if '_WT_' not in sample_id: 
        continue
    points = adata.obsm["spatial"].astype(float)

    domain = np.array([[points[:, 0].min(), points[:, 0].max()],[points[:, 1].min(), points[:, 1].max()]]) 

    pc = generatePointCloud(sample_id, points, domain=domain) 
    sen_like = adata.obs["sen_like"].astype(str).to_numpy()
    pc.addLabels("sen_like", "categorical", sen_like)

    if "1" not in pc.labels["sen_like"]["categories"]: 
        continue

    n_sen = int(np.sum(sen_like == "1"))

    r, g, _ = pairCorrelationFunction(pc, labelName="sen_like", categoriesToPlot=["1", "1"], maxR=maxR, annulusStep=annulusStep, annulusWidth=annulusWidth)
    g = np.asarray(g).flatten()
    perm_mat = np.empty((B_pcf, len(r)))
    
    for b in range(B_pcf):
        perm = rng_pcf.permutation(sen_like)
        pc_perm = generatePointCloud(sample_id, points, domain=domain)
        pc_perm.addLabels("sen_like", "categorical", perm)
    
        _, g_perm, _ = pairCorrelationFunction(pc_perm, labelName="sen_like", categoriesToPlot=["1", "1"], maxR=maxR, annulusStep=annulusStep, annulusWidth=annulusWidth)
        perm_mat[b, :] = np.asarray(g_perm).flatten()
    
    q975 = np.quantile(perm_mat, 0.975, axis=0)
    
    pcf_list.append(pd.DataFrame({
        "sample_id": sample_id,
        "r": r,
        "g": g,
        "q975": q975}))

pcf_summary = pd.concat(pcf_list, ignore_index=True)
pcf_summary = pcf_summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["g"])

wide = pcf_summary.pivot(
    index="sample_id",
    columns="r",
    values="g")


        

##SUPPLEMENTARY INFORMATION

from scipy.stats import kruskal, spearmanr

#Sample and temporal information
rows = []
for nm, sub in df_out_2[df_out_2['cond'] == 'WT'].groupby('adata'):
    day = int(nm.split('day')[1].split('_')[0]) if 'day' in nm else 0
    rows.append([nm, day])

desc = pd.DataFrame(rows, columns=['sample', 'day']).sort_values('day')

#Add PC1 scores
desc = desc.merge(meta[['adata', 'PC1']], left_on='sample', right_on='adata', how='left').drop(columns='adata')

color_map = {0: "grey", 1: "red", 7: "green", 14: "blue"}
label_map = {0: "Sham sample", 1: "Day 1 samples", 7: "Day 7 samples", 14: "Day 14 samples"}
rng_j = np.random.default_rng(0)

#PCF peak height, peak distance and aggregation radius

samples_zoom = {s: s for s in adatas if "_WT_" in s}

for lbl, s in samples_zoom.items():
    pts = adatas[s].obsm["spatial"].astype(float)
    print(lbl, np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))

maxR_long = 500          
annulusStep = annulusWidth = 2.5
B = 199
rng = np.random.default_rng(0)

pcf_long = []
for lbl, sample_id in samples_zoom.items():
    adata = adatas[sample_id]
    points = adata.obsm["spatial"].astype(float)
    domain = np.array([[points[:, 0].min(), points[:, 0].max()],
                       [points[:, 1].min(), points[:, 1].max()]])
    sen_like = adata.obs["sen_like"].astype(str).to_numpy()

    pc = generatePointCloud(sample_id, points, domain=domain)
    pc.addLabels("sen_like", "categorical", sen_like)
    r, g, _ = pairCorrelationFunction(pc, labelName="sen_like",
                                      categoriesToPlot=["1", "1"], maxR=maxR_long,
                                      annulusStep=annulusStep, annulusWidth=annulusWidth)
    g = np.asarray(g).flatten()

    perm = np.empty((B, len(r)))
    for b in range(B):
        pcp = generatePointCloud(sample_id, points, domain=domain)
        pcp.addLabels("sen_like", "categorical", rng.permutation(sen_like))
        _, gp, _ = pairCorrelationFunction(pcp, labelName="sen_like",
                                           categoriesToPlot=["1", "1"], maxR=maxR_long,
                                           annulusStep=annulusStep, annulusWidth=annulusWidth)
        perm[b] = np.asarray(gp).flatten()

    pcf_long.append(pd.DataFrame({
        "label": lbl, "sample_id": sample_id, "r": r, "g": g,
        "sim_mean": perm.mean(axis=0),
        "q975": np.quantile(perm, 0.975, axis=0)}))

pcf_long = pd.concat(pcf_long, ignore_index=True)
pcf_features = []

for nm, sub in pcf_long.groupby("sample_id"):
    sub = sub.sort_values("r")
    r = sub["r"].to_numpy(float)
    g = sub["g"].to_numpy(float)
    q975 = sub["q975"].to_numpy(float)

    keep = r >= 5

    k = np.where(keep)[0][np.nanargmax(g[keep])]
    peak_h = g[k]
    peak_r = r[k]

    above = g > q975
    r_agg = r[above].max() if above.any() else np.nan   # last distance where g(r) exceeds the envelope

    pcf_features.append([nm, peak_h, peak_r, r_agg])

pcf_features = pd.DataFrame(pcf_features, columns=["sample", "peak_h", "peak_r", "r_agg"])
desc = desc.merge(pcf_features, on="sample", how="left")


#PC1 score
plt.figure(figsize=(12, 9))

for d, g in desc.groupby("day"):
    x = np.full(len(g), d, float) + rng_j.uniform(-0.35, 0.35, len(g))
    plt.scatter(x, g["PC1"], s=80, color=color_map[d], label=label_map[d])

med = desc[desc["day"] > 0].groupby("day")["PC1"].median()
plt.plot(med.index, med.values, "k--", linewidth=2, marker="o", markersize=8, label="Median")

plt.xlabel("Time post-MI", fontsize=32)
plt.ylabel("PC1 score", fontsize=32)
plt.title("Temporal evolution of PC1 scores", fontsize=30)
plt.xticks([1, 7, 14], ["day 1", "day 7", "day 14"], fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=24)
plt.show()


#PCF peak height
plt.figure(figsize=(12, 9))

for d, g in desc.groupby("day"):
    x = np.full(len(g), d, float) + rng_j.uniform(-0.35, 0.35, len(g))
    plt.scatter(x, g["peak_h"], s=80, color=color_map[d], label=label_map[d])

med = desc[desc["day"] > 0].groupby("day")["peak_h"].median()
plt.plot(med.index, med.values, "k--", linewidth=2, marker="o", markersize=8, label="Median")

plt.xlabel("Time post-MI", fontsize=32)
plt.ylabel("PCF peak height", fontsize=32)
plt.title("Maximum aggregation intensity over time", fontsize=30)
plt.xticks([1, 7, 14], ["day 1", "day 7", "day 14"], fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=24)
plt.show()


#PCF peak distance
plt.figure(figsize=(12, 9))

for d, g in desc.groupby("day"):
    x = np.full(len(g), d, float) + rng_j.uniform(-0.35, 0.35, len(g))
    plt.scatter(x, g["peak_r"], s=80, color=color_map[d], label=label_map[d])

med = desc[desc["day"] > 0].groupby("day")["peak_r"].median()
plt.plot(med.index, med.values, "k--", linewidth=2, marker="o", markersize=8, label="Median")

plt.xlabel("Time post-MI", fontsize=32)
plt.ylabel("PCF peak distance", fontsize=32)
plt.title("Distance of maximum aggregation over time", fontsize=30)
plt.xticks([1, 7, 14], ["day 1", "day 7", "day 14"], fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=24)
plt.show()


#Aggregation radius
plt.figure(figsize=(12, 9))

for d, g in desc.groupby("day"):
    x = np.full(len(g), d, float) + rng_j.uniform(-0.35, 0.35, len(g))
    plt.scatter(x, g["r_agg"], s=80, color=color_map[d], label=label_map[d])

med = desc[desc["day"] > 0].groupby("day")["r_agg"].median()
plt.plot(med.index, med.values, "k--", linewidth=2, marker="o", markersize=8, label="Median")

plt.xlabel("Time post-MI", fontsize=32)
plt.ylabel("Aggregation radius", fontsize=32)
plt.title("Spatial extent of aggregation over time", fontsize=30)
plt.xticks([1, 7, 14], ["day 1", "day 7", "day 14"], fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=24)
plt.show()

#Statistical tests
rows = []

for v in ["PC1", "peak_h", "peak_r", "r_agg"]:
    d = desc[desc["day"] > 0].dropna(subset=[v])
    H, p_kw = kruskal(*[g[v].values for _, g in d.groupby("day")])
    rho, p_sp = spearmanr(d["day"], d[v])
    rows.append([v, len(d), H, p_kw, rho, p_sp])

res = pd.DataFrame(rows, columns=["variable", "n", "H", "p_kruskal", "rho", "p_spearman"])

print(desc[["sample", "day", "PC1", "peak_h", "peak_r", "r_agg"]])
print(res.round(3))

rows = []

for sample, adata in adatas.items():
    if "_WT_" not in sample:
        continue
    n_total = adata.n_obs
    n_sen = int(adata.obs["sen_like"].astype(int).sum())
    rows.append([sample, n_total, n_sen, 100*n_sen/n_total])

check_samples = pd.DataFrame(rows, columns=["sample", "n_total_spots", "n_sen_like", "percent_sen_like"])
print(check_samples.to_string(index=False))

#Influence of day14_1 on the aggregated curves (Fig 2d)
cible = "GSM5355668_WT_MI_day14_1"
wt = df_out_2[df_out_2["cond"] == "WT"]

avec = wt.groupby("r_value")["L_stat"].agg(["mean", "median"])
sans = wt[wt["adata"] != cible].groupby("r_value")["L_stat"].agg(["mean", "median"])

plt.figure(figsize=(12, 8))
plt.plot(avec.index, avec["mean"], color="black", linewidth=2, label="Mean (n=10)")
plt.plot(sans.index, sans["mean"], color="red", linewidth=2, label="Mean without Day 14 (Rep. 1), n=9")
plt.plot(avec.index, avec["median"], color="black", linestyle="--", linewidth=2, label="Median (n=10)")
plt.plot(sans.index, sans["median"], color="red", linestyle="--", linewidth=2, label="Median without Day 14 (Rep. 1), n=9")
plt.xlabel("Distance r", fontsize=31)
plt.ylabel("L(r)", fontsize=31)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=22)
plt.show()

for stat in ["mean", "median"]:
    d = (sans[stat] - avec[stat]) / avec[stat] * 100
    print(f"{stat}: {d.mean():.1f}% on average, max {d.max():.1f}%")
    
    
#Influence of day14_1 on the FPCA (Fig 4a)
noms = [n for n in df_out_2["adata"].unique()]
courbes9, r9 = [], None
for nm in noms:
    if nm == cible:
        continue
    t = df_out_2[(df_out_2["adata"] == nm) & (df_out_2["r_value"] >= 40)].sort_values("r_value")
    r9 = t["r_value"].values
    courbes9.append((t["L_stat"] - t["sim_mean"]).values)

fd9 = FDataGrid(data_matrix=courbes9, grid_points=r9)
fd9s = BasisSmoother(basis=BSplineBasis(domain_range=(r9.min(), r9.max()), n_basis=20)).fit_transform(fd9)
fpca9 = FPCA(n_components=2)
fpca9.fit(fd9s)

print("variance explained with n=10:", np.round(fpca.explained_variance_ratio_, 4))
print("variance explained with n=9 :", np.round(fpca9.explained_variance_ratio_, 4))

plt.figure(figsize=(12, 8))
plt.plot(r9, fpca.components_[0](r9)[0], color="black", linewidth=2, label="PC1 (n=10)")
plt.plot(r9, fpca9.components_[0](r9)[0], color="red", linewidth=2, label="PC1 without Day 14 (Rep. 1), n=9")
plt.xlabel("Distance r", fontsize=31)
plt.ylabel("Component value", fontsize=31)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.show()


#Sensitivity of the L(r) FPCA to smoothing parameters
courbes_s, r_s = [], None

for nm in df_out_2["adata"].unique():
    t = df_out_2[(df_out_2["adata"] == nm) & (df_out_2["r_value"] >= 40)].sort_values("r_value")
    r_s = t["r_value"].values
    courbes_s.append((t["L_stat"] - t["sim_mean"]).values)

courbes_s = np.array(courbes_s)

results_L = []

for rmax_test in [150, 200, 250]:
    mask = r_s <= rmax_test
    for K in [10, 15, 20, 25]:
        fd = FDataGrid(data_matrix=courbes_s[:, mask], grid_points=r_s[mask])
        fds = BasisSmoother(basis=BSplineBasis(domain_range=(r_s[mask].min(), r_s[mask].max()), n_basis=K)).fit_transform(fd)
        fp = FPCA(n_components=2)
        fp.fit(fds)
        results_L.append([rmax_test, K, fp.explained_variance_ratio_[0]*100, fp.explained_variance_ratio_[1]*100])

results_L = pd.DataFrame(results_L, columns=["r_max", "n_basis", "PC1", "PC2"])
print(results_L.round(2).to_string(index=False))


hub_cols = [c for c in adatas[list(adatas)[0]].obs.columns if c.endswith("_out")]
hub_names = [c.replace("_out", "") for c in hub_cols]

threshold_results = []

for k in [2, 2.5, 3, 3.5, 4]:
    for sample, adata in adatas.items():
        if "_WT_" not in sample:
            continue

        labels = []
        for hub in hub_names:
            s = adata.obs[hub]
            labels.append((s > s.mean() + k*s.std()).astype(int))

        sen_like_k = (np.sum(labels, axis=0) > 0).astype(int)
        threshold_results.append([k, sample, len(sen_like_k), sen_like_k.sum(), 100*sen_like_k.mean()])

threshold_results = pd.DataFrame(threshold_results, columns=["threshold", "sample", "n_total", "n_sen", "percent_sen"])
print(threshold_results.to_string(index=False))

#Sensitivity of Ripley's L(r) to the binarization threshold
ripley_threshold = []

for k in [2, 2.5, 3, 3.5, 4]:
    for sample, adata in adatas.items():
        if "_WT_" not in sample:
            continue

        labels = []
        for hub in hub_names:
            s = adata.obs[hub]
            labels.append((s > s.mean() + k*s.std()).astype(int))

        adata.obs["sen_like_threshold"] = pd.Categorical((np.sum(labels, axis=0) > 0).astype(int))

        ripley_result = sq.gr.ripley(adata, cluster_key="sen_like_threshold", mode="L", max_dist=250, n_steps=100, copy=True)
        ripley_df = ripley_result["L_stat"]
        ripley_df = ripley_df[ripley_df["sen_like_threshold"] == 1].reset_index(drop=True)

        for j in range(len(ripley_df)):
            ripley_threshold.append([k, sample, ripley_df.loc[j, "bins"], ripley_df.loc[j, "stats"]])

        del adata.obs["sen_like_threshold"]

ripley_threshold = pd.DataFrame(ripley_threshold, columns=["threshold", "sample", "r_value", "L_stat"])

plt.figure(figsize=(13, 8))

for k in [2, 2.5, 3, 3.5, 4]:
    temp = ripley_threshold[ripley_threshold["threshold"] == k].groupby("r_value")["L_stat"]
    median_L = temp.median()
    plt.plot(median_L.index, median_L.values, linewidth=2.5, label=fr"Threshold: $\mu+{k}\sigma$")

plt.xlabel("Distance r", fontsize=31)
plt.ylabel("Median L(r)", fontsize=31)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=20)
plt.show()



hub_cols = [c for c in adatas[list(adatas)[0]].obs.columns if c.endswith("_out")]

#Strict rule: global threshold AND sample-specific threshold (the _out columns)
print("=== STRICT (combined threshold) ===")
for sample, adata in adatas.items():
    if "_WT_" not in sample:
        continue
    print("\n", sample)
    for hub in hub_cols:
        print(hub.replace("_out", ""), ":", int(adata.obs[hub].astype(int).sum()))

#Looser rule: sample-specific threshold only (the one used in the analyses)
print("\n\n=== LOOSE (sample-specific threshold) ===")
for sample, adata in adatas.items():
    if "_WT_" not in sample:
        continue
    print("\n", sample)
    for hub in hub_cols:
        h = hub.replace("_out", "")
        s = adata.obs[h]
        print(h, ":", int((s > s.mean() + 3 * s.std()).sum()))
        

##### Cross-PCF analysis

import sys
sys.path.append(r"C:\Users\cecil\Thèse  partie stat")

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import cdist
from helperFunctions import generatePointCloud, pairCorrelationFunction, crossPCF, getAnnulusAreasAroundPoints

MIN_SPOTS = 10
MIN_SAMPLES_PAIR = 5
maxR = 250
annulusStep = annulusWidth = 2.5
B_perm = 99
rng_i = np.random.default_rng(0)

hub_names = [c.replace("_out", "") for c in hub_cols]
wt_samples = [s for s in adatas if "_WT_" in s]

# Hub-specific binary labels
for sample in wt_samples:
    adata = adatas[sample]
    for hub in hub_names:
        score = adata.obs[hub]
        adata.obs[hub + "_loose"] = (score > score.mean() + 3*score.std()).astype(int)

# Hubs usable for intra-hub PCF
usable_hubs = []
for hub in hub_names:
    n_usable = sum(adatas[s].obs[hub + "_loose"].sum() >= MIN_SPOTS for s in wt_samples)
    if n_usable >= MIN_SAMPLES_PAIR:
        usable_hubs.append(hub)

# Pairs usable for cross-PCF
usable_pairs = []
for hubA, hubB in combinations(hub_names, 2):
    n_usable = sum((adatas[s].obs[hubA + "_loose"].sum() >= MIN_SPOTS) and (adatas[s].obs[hubB + "_loose"].sum() >= MIN_SPOTS) for s in wt_samples)
    if n_usable >= MIN_SAMPLES_PAIR:
        usable_pairs.append((hubA, hubB))

# Intra-hub PCF
pcf_intra = []

for sample in wt_samples:
    adata = adatas[sample]
    points = adata.obsm["spatial"].astype(float)
    domain = np.array([[points[:, 0].min(), points[:, 0].max()], [points[:, 1].min(), points[:, 1].max()]])
    n = len(points)

    for hub in usable_hubs:
        label = adata.obs[hub + "_loose"].to_numpy()
        if label.sum() < MIN_SPOTS:
            continue

        pc = generatePointCloud(sample, points, domain=domain)
        pc.addLabels("hub", "categorical", label.astype(str))
        r, g, _ = pairCorrelationFunction(pc, labelName="hub", categoriesToPlot=["1", "1"], maxR=maxR, annulusStep=annulusStep, annulusWidth=annulusWidth)
        g = g.flatten()

        perm = np.empty((B_perm, len(r)))
        for i in range(B_perm):
            lp = np.zeros(n, int)
            lp[rng_i.choice(n, int(label.sum()), replace=False)] = 1
            pcp = generatePointCloud(sample, points, domain=domain)
            pcp.addLabels("hub", "categorical", lp.astype(str))
            _, gp, _ = pairCorrelationFunction(pcp, labelName="hub", categoriesToPlot=["1", "1"], maxR=maxR, annulusStep=annulusStep, annulusWidth=annulusWidth)
            perm[i] = gp.flatten()

        sim_mean = perm.mean(axis=0)

        for j, ri in enumerate(r):
            pcf_intra.append([sample, hub, int(label.sum()), ri, g[j], sim_mean[j]])

pcf_intra = pd.DataFrame(pcf_intra, columns=["sample", "hub", "n_spots", "r", "g", "sim_mean"])

# Intra-hub PCF figure
hubs_plot = sorted(pcf_intra["hub"].unique())
couleurs = plt.cm.tab10(np.linspace(0, 1, len(hubs_plot)))

fig, axes = plt.subplots(3, 2, figsize=(24, 22))
axes = axes.flatten()

for i, (h, c) in enumerate(zip(hubs_plot, couleurs)):
    sub = pcf_intra[(pcf_intra["hub"] == h) & (pcf_intra["r"] <= 120)]
    obs = sub.groupby("r")["g"].median()
    csr = sub.groupby("r")["sim_mean"].median()
    n_s = sub["sample"].nunique()

    ax = axes[i]
    ax.plot(obs.index, obs.values, color=c, linewidth=2, marker="o", markersize=5, label="Observed PCF")
    ax.plot(csr.index, csr.values, color="black", linewidth=2, linestyle="--", label="Mean CSR")
    ax.set_xlabel("Distance r", fontsize=29)
    ax.set_ylabel("PCF g(r)", fontsize=29)
    ax.set_title(h.replace("Heart_and_Aorta__", "").replace("__", " ") + f" (n={n_s})", fontsize=30)
    ax.tick_params(axis="both", labelsize=27)
    ax.legend(fontsize=27, loc="upper right")

for j in range(len(hubs_plot), len(axes)):
    axes[j].axis("off")

plt.tight_layout(h_pad=3)
plt.savefig("intra_hub_PCF.png", dpi=300, bbox_inches="tight")
plt.show()

# Cross-hub PCF
pcf_cross = []

for sample in wt_samples:
    adata = adatas[sample]
    points = adata.obsm["spatial"].astype(float)
    domain = np.array([[points[:, 0].min(), points[:, 0].max()], [points[:, 1].min(), points[:, 1].max()]])
    total_area = (domain[0, 1] - domain[0, 0]) * (domain[1, 1] - domain[1, 0])
    n = len(points)

    for hubA, hubB in usable_pairs:
        maskA = adata.obs[hubA + "_loose"].to_numpy() == 1
        maskB = adata.obs[hubB + "_loose"].to_numpy() == 1
        nA, nB = int(maskA.sum()), int(maskB.sum())

        if nA < MIN_SPOTS or nB < MIN_SPOTS:
            continue

        pointsA = points[maskA]
        areasA = getAnnulusAreasAroundPoints(pointsA, maxR, annulusStep, annulusWidth, domain)
        r, g, _ = crossPCF(cdist(pointsA, points[maskB]), areasA, nB/total_area, maxR, annulusStep, annulusWidth)
        g = g.flatten()

        perm = np.empty((B_perm, len(r)))
        for i in range(B_perm):
            idx = rng_i.choice(n, nB, replace=False)
            _, gp, _ = crossPCF(cdist(pointsA, points[idx]), areasA, nB/total_area, maxR, annulusStep, annulusWidth)
            perm[i] = gp.flatten()

        sim_mean = perm.mean(axis=0)

        for j, ri in enumerate(r):
            pcf_cross.append([sample, hubA, hubB, nA, nB, ri, g[j], sim_mean[j]])

pcf_cross = pd.DataFrame(pcf_cross, columns=["sample", "hub_A", "hub_B", "n_A", "n_B", "r", "g", "sim_mean"])

print("Usable pairs:", len(usable_pairs))
print(usable_pairs)

# Cross-hub PCF figure
pairs_plot = sorted(set(zip(pcf_cross["hub_A"], pcf_cross["hub_B"])))
couleurs_c = plt.cm.tab10(np.linspace(0, 1, len(pairs_plot)))

fig, axes = plt.subplots(4, 2, figsize=(24, 29))
axes = axes.flatten()

for i, ((hA, hB), c) in enumerate(zip(pairs_plot, couleurs_c)):
    sub = pcf_cross[(pcf_cross["hub_A"] == hA) & (pcf_cross["hub_B"] == hB) & (pcf_cross["r"] <= 120)]
    obs = sub.groupby("r")["g"].median()
    csr = sub.groupby("r")["sim_mean"].median()
    n_s = sub["sample"].nunique()
    nom = hA.replace("Heart_and_Aorta__", "").replace("__", " ") + " - " + hB.replace("Heart_and_Aorta__", "").replace("__", " ")

    ax = axes[i]
    ax.plot(obs.index, obs.values, color=c, linewidth=2, marker="o", markersize=5, label="Observed cross-PCF")
    ax.plot(csr.index, csr.values, color="black", linewidth=2, linestyle="--", label="Mean CSR")
    ax.set_xlabel("Distance r", fontsize=31)
    ax.set_ylabel("Cross-PCF g(r)", fontsize=27)
    ax.set_title(f"{nom} (n={n_s})", fontsize=29)
    ax.tick_params(axis="both", labelsize=27)
    ax.legend(fontsize=25, loc="upper right", bbox_to_anchor=(0.98, 1.18), frameon=False)

for j in range(len(pairs_plot), len(axes)):
    axes[j].axis("off")

plt.tight_layout(h_pad=3)
plt.show()



