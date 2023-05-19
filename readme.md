This respository contains code to produce geographical simulations of Cornish and Welsh over the time periods considered in the study.

Contents:

- `/code`: Python code used for the experiment including generating simulations, extracting data of interest such as population/area coverage and plotting data. The purpose of the different files is detailed below but as a summary:
    - `smooth_population.py` creates smoothed population from Output Areas.
    - `simulate_{country}.py` generates simulations for the respective regions.
    - `save_times_population_and_areas.py` creates the files named `times_....npy`, `area1....npy` and `pop1....npy` from the raw simulations containing the timesteps and the area and population speaking the particular language of interest at each step. These are saved in `/data`.  
    - `{country}_pop.py` and `{country}_area.py` use these files to create the figures showing population and area over time shown in the paper.
    - `snapshots{country}_paper.py` creates the snapshots of the simulations shown in the paper.
    - `welshrealdata.py` and `comparing_wales.ipynb` create figures for the real Welsh data and walks through the creation of the figure showing spatial comparison between this and the simulation (Fig. 6).


- `/assets`: Shape files in either `.npy` or `.tif` format required for running the simulations.
    - Files named `{country}_mask` cover the whole of the region simulated. For example, `wales_mask.npy` covers much of mid-western England.
    - Files named `{country}_country_mask` contain the shape of the country or county for the purposes of calculating population and area. These are obtained from the UK Gov website.
    - The initial conditions are binary masks named `cornwall_river_mask.npy` and `wales_initial_1850.npy` (the names referring to the descriptions of the initial conditions used; see paper for discussion)
    - Files named `{country}_pop_dist` contain the population for each Output Area in the map (as discussed below)
    - Files named `{country}_smoothed_dist_ss{x}.npy` are generated using `smooth_population.py` and contain the population distribution used when generating simulations.

-`/data`: Contains the computed times (`times....npy`), areas (`area1....npy`) and population (`pop1....npy`) data used in the report. Other raw data can be generated using `simulate_{country}.py`. The format for these is:
`cornwallPopGaussian{sigma_smooth}ICriverAlpha{alpha}Beta{beta}SigmavarFactor1Deltat{delta_t}Tmax{tmax}_{folder_num}`. `ICriver` in the Cornwall files and `ICbook` in the Wales files refer to the use of the initial conditions as described in the Methods section, either being the river boundary or taken from Pryce (1978). The roles of the other parameters are best understood by reading the Theory section of the paper. `Sigmavar` refers to the use of a $\sigma$ dependent on the population density, and `Factor1` refers to the use of $log(2)/10$ as a scaling factor.

The pipeline of the experiments is as follows:
1. Obtain shapefiles of regions and census Output Areas from UK Gov website. For example, `wales_mask.npy` (`cornwall_mask.tif`) contains the shape for Wales (Cornwall) and neighbouring parts of England (the binary mask used to determine the region over which the simulation is performed) and `wales_pop_dist.npy` (`cornwall_pop_dist.npy`) contains the population for each region (for methodological reasons each pixel of the element in the array contains the population of the region that element is in). 
2. Following the method from Burridge, smooth the population distribution by replacing the population in each Output Area with a Gaussian distribution, with population falling outside the region iteratively redistributed so the total population is preserved. This is done using `smooth_population.py` and the resulting file is called `wales_smoothed_dist_ss10.npy` (`ss` referring to σ_s).
3. The simulation carried out with `simulate_wales.py` (`simulate_cornwall.py`), using the population distribution generated above, the initial condition `wales_initial_1850.npy` (`cornwall_river_mask.npy`) and the region masks. The initial conditions were manually created Snapshots are saved in an automatically generated folder every `saveInterval` iterations. The simulation can be very intensive depending on hyperparams, timestep etc. (HPC was used to create the figures in the report).
4. The timestamps, areas and populations are extracted from the simulated data using `save_times_population_and_area.py` and are stored in the files named `times_...`, `area1...`, `pop1...` ('1' referring to 'variant 1', i.e. the language of interest).
5. Figures are plotted and comparisons made to historical data using the various files in the repo (`/data`). Other masks (e.g. `cornwall_mask_w_border.npy` and `wales_country_mask_tif.tif`) are used for creating figures.
