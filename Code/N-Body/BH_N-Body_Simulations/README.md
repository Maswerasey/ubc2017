This folder contains code to examine the N_body simulation data used in:
"An intermediate-mass black hole in the centre of the globular cluster 47 Tucanae" Nature paper.
https://arxiv.org/abs/1702.02149


model_code.py produces v(r) plots and uses the maximum likelihood function fitting for the plummer model
 to regain the black hole mass. 

Note: resulting Mass ratio is very different from the one given in the paper.

Nbody_king contains the King model analysis of the Nbody simulations. 
However, this code contains bugs and gives wrong results (0%BH for all models). 
Also the problem about which tidal radius should be used remains unsolved. 
