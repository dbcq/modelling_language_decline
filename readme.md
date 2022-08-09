This respository contains code to produce geographical simulations of Cornish and Welsh over the time periods considered in the study.

The pipeline is as follows:
1. Obtain shapefiles of regions and census Output Areas from UK Gov website. For example, `wales_plus_mask.npy` contains the shape for Wales (the binary mask used to determine the region over which the simulation is performed) and __ contains the population for each region. 
2. Following the method from Burridge, smooth the population distribution by replacing the population in each Output Area with a Gaussian distribution, with population falling outside the region iteratively redistributed so the total population is preserved.
3. The simulation is then carried out with `calculatemcountrymask_single`, using the population distribution generated above and the region masks.