# ml-hardness-vickers-thesis


This is my thesis project, I need to predict the Hardness Vickers for ...

## DATA PREPARATION (Manually)
Using the global Database in HardnessVickersDatabase.xlsx we get the following columns: 

*Reference*: This column is used to identify from where this data comes and is not useful to train the model.
...
...
*Hardness (HV)*: This is the meassured hardness for all the inputs.

Saved the .xslx as a .csv in order to be compatible with pandas processing tools.

I cut of the file the Reference column to avoid any bias or noise in the training.

removing special character out from the column names to get a development-friendly column names

`Ti_at,Nb_at_,Hf_at_,Zr_at_,Al_at_,Ta_at_,Mo_at_,W_at_,V_at_,Cr_at_,Si_at_,Processing_Temperature_C_,Treatment_Temperature_C_,Treatment_Time_h_,Average_atomic_radius_pm_,Atomic_radius_difference_pm_,Valence_Electron_Concentration,Enthalpy_of_Mixing_J_mol_,Mixing_Entropy_J_K_mol_,Hardness_HV_`

aboves shows the post edited column names which are present in HardnessVickersDataset.csv

We're ready to start the job.

## PREPARING ENVIRONMENT

I used python 3.12 to build this project.
`https://www.python.org/downloads/`

Create a virtual environment first (Yes, is a best practice to have one environment per project)

`python3.12 -m venv .venv`

then activate it 

`source .venv/bin/activate`

the terminal should change with the name of the environment.

after that, install the requirements. I used pip to handle the reqs.

`pip install -r requirements.txt`

You're all set!

## DATA PREPARATION (Auto using pandas)

