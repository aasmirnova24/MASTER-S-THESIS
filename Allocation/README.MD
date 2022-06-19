# Collection Allocation
Allocation - a model for assessing the probability of receiving payments from a client in arrears. This model allows us to effectively manage the strategy for collecting overdue debts. The statistical unit is the customer who is 31 days past due. This model is a binary classification.

## Target
- “1” - after the delay, the client makes payments, 
- “0” - after the delay, the client did not make payments and the number of days in delay exceeded 60 days.

## DataSet
- Observations: ***256 220***		
- Features: ***162***
- Minority class percentage (bad-rate): ***37,19%***

## Python-notebooks:
- Feature selection [Full CFSS](https://github.com/aasmirnova24/MASTER-S-THESIS/blob/main/Allocation/2022_Dip_Allocation_all%20select.ipynb)
- Experiments [Experiments 1 -2](https://github.com/aasmirnova24/MASTER-S-THESIS/blob/main/Allocation/2022_Dip_Allocation_experiment%201-2.ipynb)
