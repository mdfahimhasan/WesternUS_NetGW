# A Data-driven Approach for Estimating Effective Precipitation in the Western United States: Applications in Irrigation Groundwater Management

## Abstract
Effective precipitation, defined as the portion of ET derived from precipitation, is a crucial component of the agricultural water balance and controls the amount of water required for irrigation. Quantification and validation of effective precipitation is challenging as it comprises of multiple fluxes – precipitation, surface runoff, and deep percolation. Existing empirical and process-based effective precipitation estimation approaches lack complexity, accuracy, and validation, especially for large-scale applications.  Moreover, there is no readily available high-resolution effective precipitation dataset for the United States (US), despite its importance in water management. In this study, we focused on the Western United States (Western US), a region severely impacted by excessive groundwater withdrawal, and developed a data-driven approach for estimating effective precipitation at high spatio-temporal resolution. The framework incorporates a machine learning model that extracts effective precipitation information from non-irrigated croplands and provides its estimates for the irrigated croplands of the Western US at a 2 km resolution and on a monthly scale from 2000 to 2020. The model performs satisfactorily, with a test R2 of 0.831 and an RMSE of 11.23 on the holdout set. We analyzed the factors influencing effective precipitation to understand its dynamics in irrigated landscapes and identified maximum relative humidity, precipitation, and reference evapotranspiration as some of the key driving factors. To validate the effective precipitation estimates, we measured groundwater pumping over seven basins of the Western US using a water balance approach that incorporates the model-generated effective precipitation estimates. A comparison of our estimated pumping with in-situ records shows R2 of 0.87 and 0.75 in volumetric and depth spaces, respectively. The successful implementation and satisfactory performance of our model and estimated pumping values show the potential of our effective precipitation estimation approach in irrigation water management. Furthermore, the high-resolution effective precipitation dataset will play a pivotal role in satellite-based groundwater pumping monitoring, thereby supplementing in-situ groundwater monitoring efforts. 

__Keywords:__ Effective precipitation, Groundwater; Irrigation; Water use; Remote sensing; Machine learning.

## Effective precipitation map
<img src="readme_figs/Peff_map.png" height="500"/> 

Figure: Machine learning model generated monthly and growing season effective precipitation estimates for 2016 at 2 km spatial resolution.

## Publications
in progress

## Organizations
<img src="readme_figs/CSU-Signature-C-357-617.png" height="90"/> <img src="readme_figs/Official-DRI-Logo-for-Web.png" height="80"/>

## Funding
<img src="readme_figs/NASA-Logo-Large.png" height="80"/>

## Running the repository

### Repository structure
The repository has five main modules described as follows-

__1. utils -__ consists of scripts that helps in basic raster, vector, and statistical operation. It also holds the `ml_ops` scripts which has the machine learning functions.

__2. data_download_preprocess -__ consists of scripts that have functions to download datasets from GEE, including OpenET, and to further pre-process the datasets. The `run_download_preprocess.py` is the main driver script that has to be used to download and pre-process all required datasets.

__3. effective_precip -__ consists of functions that are required specifically for the effective precipitation model. The effective precipitation is estimated by a 3-step model. First, the `m01_peff_model_monthly.py` script estmates effective precipitation at monthly scale. The monthly estimates do not follow water balance (water year precipitation > water year effective precipitation) in some regions. So, at the second step,  the `m02_peff_frac_model_water_yr.py` script simulates a water year-scale effective precipitation fraction model. This water year-scale model is used to impose water balance over the monthly estimates using the `m03_peff_adjust.py` script. These three files have to be run in sequence to generate the monthly effective precipitation estimates.

__4. sw_irrig -__ consists of functions that are required for dictributing USGS HUC12 level surface water irrigation data to 2 km pixel scale. The `SW_Irr.py` is the main driver file.

__5. netGW -__ consists of the `netGW_Irr.py` script that has the functions to estimate consumptive groundwter use for irrigation at 2 km resolution using a water balance appraoch.  

The __utils__ module do not need any execution. The latter modules are required to be executed using the respective driver files to unvail the full funtionality of the model. The repository has other auxiliary folders with scripts that are used for some data processing, result analysis,and plotting purposes.

### Dependencies
__operating system:__ Most scripts are fully functional in windows and linux environments except some. In linux environment, gdal needs to be installed separately and the appropriate 'gdal_path' needs to be set in necessary scripts. For some functions, e.g. the `shapefile_to_raster()` in `utils > raster_ops.py` and associated scripts (`results_analysis > netGW_pumping_compile.py`), gdal system call has to enabled/installed specifically to run them in linux environment. Note that all scripts, except the scripts in `results_analysis` module, have been implemented/checked using both windows and linux environment (using conda environment). In addition, the ALE plot generation in `m01_peff_model_monthly.py` and `m02_peff_frac_model_water_yr.py` scripts do not respond (keep running indifinitely) in linux environment (probably due to scikit-explain versioning issue); therefore, set `skip_plot_ale = True` when running the monthly and water year models in linux environment.

The authors recommend exercising discretion when setting up the environment and run the scripts.

__conda environment:__ A _conda environment_, set up using [Anaconda](https://www.anaconda.com/products/individual) with python 3.9, has been used to implement this repositories. Required libraries needed to be installed to run this repository are - dask, dask-geopandas, earthengine-api, fastparquet, rasterio, gdal, shapely, geopandas, numpy, pandas, scikit-learn, lightgbm, scikit-explain, matplotlib, seaborn. 

Note that running the `.ipynb` scripts will require installaion of jupyter lab within the conda environment.

### Google Earth Engine authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) some of the predictor datasets from the GEE
data repository. After completing step 3, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). The Google Cloud CLI tools
may be required for this GEE authentication step. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk). You also have to create a gcloud project to use the GEE API. 


## Data availability
The monthly effective precipitation estimates for all months of 2000 to 2020 (up to September in 2020) are available to download through this [GEE script](https://code.earthengine.google.com/8a41d8cbfb32e73cb339be98c111c875). Non-GEE users can acccess the dataset from this [HydroShare repo](https://www.hydroshare.org/resource/c33ce80f5ae44fe6ab2e5dd3c128eb0b/).

