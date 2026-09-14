# Senescence spatial analysis

This repository contains the Python code associated with the manuscript:

**“Multiscale Analysis of Cellular Senescence through Ripley’s Functions and Functional Statistics”**

The study investigates the spatial organization of senescence-associated transcriptional programs in mouse heart tissue following myocardial infarction. Spatial point-pattern statistics and functional data analysis are used to characterize aggregation patterns across tissue sections and spatial scales.

## Data

The analyses are based on the publicly available spatial transcriptomics dataset **GSE176092**, described by Yamada et al.

GEO accession:  
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176092
The spatial analyses presented in the manuscript focus on the 10 wild-type samples corresponding to Sham and Days 1, 7 and 14 after myocardial infarction.
Raw expression matrices, spatial coordinates and histological images can be downloaded directly from GEO.

## Code

The script performs the complete analysis workflow used in the manuscript and Supplementary Information.

It includes preprocessing of the Visium spatial transcriptomics data with Scanpy, computation of SenePy senescence scores, identification of senescence-like spots and spatial characterization using Ripley’s L function, Moran’s I, pair-correlation functions and topographical correlation maps.

The script also contains the functional data analysis of the spatial profiles using B-spline smoothing and functional principal component analysis, together with bootstrap procedures used to assess the stability of the results.

Additional analyses included in the script correspond to the revisions and Supplementary Information, including temporal comparisons between Sham and post-infarction time points, PCF-derived spatial descriptors, sensitivity analyses, assessment of the influence of the atypical Day 14 sample, and hub-specific intra-hub and cross-hub PCF analyses. 

### Senescence scoring

SenePy mouse Heart and Aorta transcriptional hubs are used to calculate continuous senescence-associated scores for each spatial spot.
Binary senescence-like labels are subsequently derived from these scores and used for the spatial point-pattern analyses.

### Ripley’s L function

Ripley’s L function is used to characterize the spatial distribution of senescence-like spots across increasing spatial distances.
Permutation-based reference distributions are generated for comparison with complete spatial randomness.

### Pair-correlation function

Pair-correlation functions are used to characterize spatial interactions at individual distances and complement the cumulative information provided by Ripley’s L function.

### Topographical correlation maps

Topographical correlation maps are used to localize the regions of the tissue contributing to the observed spatial interactions. These maps are also superimposed on the corresponding histological H&E images.

### Functional data analysis

Spatial statistic curves are treated as functional observations. B-spline representations and functional principal component analysis are used to summarize the main modes of variation between samples.
Bootstrap resampling is used to evaluate the stability of the functional results.

## Usage

Download the GSE176092 dataset from GEO.
In `code_senescence.py`, replace the local data path with the directory containing the downloaded files.
The script should then be executed sequentially, as several analyses use objects generated in previous sections.

## Authors

Cécile Verrier  
Vanessa Dehennaut  
Sophie Dabo-Niang
