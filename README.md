v0.1.1:
To do: 
1. Display fit parameters and quality of fit values
2. Create procedure that created this notebook
3. See how to manually add fit priors. Currently set to auto.
4. Any other demands

To run: Run all in notebook. 

Data files not included. 

Known issues:
1. Fitting does not rewrite the results file... 
2. Summary plot is inefficient. Unsure if inefficiency will be resolved any time soon. It works, but just slow. 
3. Critical memory leak issue found. Looping over many parameters causes memory leaks. Attempting fix. 

Needed Testing: 
1. Run the loop for parameters and compare with output of equivalent bash script. Expected to take hours -- overnight. Will do tonight 04/04. CRITICAL ISSUE -- MEMORY LEAK. 
2. There was something else to test, but it is late and I forgot. 

v0.0.1:

To do: 
1. Clean up text outputs
2. Display fit parameters and quality of fit values
3. Create procedure that created this notebook
4. See how to manually add fit priors. Currently set to auto.
5. Clean up the parameter blocks to reduce clutter
6. Any other demands

To run: 
Run all in notebook. 

Data files not included. 

Known issues: 
1. Fitting does not rewrite the results file, so if you run the fit for 1 specific irrep, it will not run for other irreps. An error will pop up that states "remake result/deuteron_Zjn_tNorm3_evp.h5 with all irreps".
   Current bandaid fixes:
   a. Delete result folder, and rerun the code with the desired irrep to fit. Must be repeated everytime. 
   b. Delete the result folder, and run the fit for all irreps then swap to plotting for 1. 
