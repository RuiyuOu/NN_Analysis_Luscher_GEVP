v0.2.1:
To do:
1. Allow changeable plot axis limits in the notebook.
2. Test bootstrapping
3. Recode the fitting to dump fit and necessary data into separate folders for separate irreps
4. See if it is possible to toggle boosted irrep states for fitting
5. Create procedure that created this notebook
6. See how to manually add fit priors. Currently set to auto.
7. Implement fix for known issues 1
8. Any other demands

To run: Run all. You can collapse most --really all-- of the cells. If on browser localhost, click bluebar on the left of the cell to collapse it. If on vs code, double click blank area to the left of cell and below run arrow to collapse it. All the functionality is in the dropdown panels and buttons, not much is needed to be changed in the code. 

Only 1 specific fit can be done at a time on the current version. 

Data filed not included. 

Known issues: 
1. Fitting does not rewrite the results file, so if you run the fit for 1 specific irrep, it will not run for other irreps. An error will pop up that states "remake result/deuteron_Zjn_tNorm3_evp.h5 with all irreps".
   Thus the displaying of the fit parameters will come up blank for fits not done. 

Possible issues: 
1. Anomalous memory leak with vs code happened once during testing. Unable to reproduce the error. May not exist in the pushed version. 