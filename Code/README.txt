Tuc_Data contains cleaned data of 47 Tuc Main sequence, white dwarf band and giant branch. 
analysis.py produces Tuc_Data, plots, and plummer-model analysis (binned and unbinned)

BH_N-Body_Simulations Folder contains N_Body simulations from Holger B. (author of nature paper about 47 Tuc IMBH)
and analysis using maximum likelihood.
(Plummer model analysis is done, King in progress)

PLUMMER

plummer_center folder contains (plummer model) analysis to fit the centre of the cluster using the max. likelihood method
(result only slightly different, no major impact on BH-mass results)

plummer_random_lnL contains the code and results to produce random fake clusters that contain a 0% and a best fitting 
BH for 47tuc and refitting it with the model to detect biasses and error margins on the results produced. 
the important files are contained in the final_run_data_result subfolder. 

position-errors and the subfolder position_erors_fit contains bootstraped trials to deternine the real position of the cluster centre. 
important files are found in Data_results_plots


KING-MODEL
a_differenetiation: contains attempt to include 2 different a's into the king model fit. (unfinished)

King_bootstrap+random: contains bootstrap data(N= 100000 of 47 tuc using single "a" king model) in King_bootstrap_result
needs evaluation.
--King_random_result: result of generating 100000 random 47 tuc like clusters wit no/best fit BH and refitting using singal "a" king model
--King_resample_cluster: code used on computer cluster to produce resampling. 

King_model_chris_data: contains Swantje's code utilizing chris' data to compare results with each other. 

files: 
king-model: produces king model analysis using single "a" model. 
fitfunc_params: parameters to approximate king model velocity dispersion. 
king_params: max. likelihood parameters from fitting king-approximation fuction to 47 tuc. 


