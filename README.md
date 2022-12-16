
######### How to run the code ############

# PART 1

1. Run "python test.py"
2. For all methods the condition is True. If you want to run for specific helper method then make the other varibles false. 


# Part 2 

1. generate_bn.py is used to create random baysian networks.
2. folder testing->Part_2 consists of random baysian network part 2 was tested on. 
3. to run the script to get the runtime of MAP and MPE for 3 different heuristics, run "python Part2.py" which will dump 2 excel in output folder. 

# Part 3 

1. run "python use_case.py"





# Useful Pointers for Assignment 2 of KR21
## BIFXML file format
The BIFXML file format is meant to provide an easy means of exchanging Bayesian networks. It works with standard XML
tags. The detailed description of the format can be found at 
http://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/. Please read this carefully. 
Note that for our purposes it will be enough to only have nodes of type "nature", and we will not need the `<Property>`
tag. 

Be aware of the order of the values in the probability table tag. They should be ordered in a way that corresponds to 
boolean counting where the "For" variable is the least significant bit followed by the "Given" variables from bottom
to top. Here an example of what this is supposed to mean. Let's assume we have the following CPT table in the BIFXML file:
```
<DEFINITION>
	<FOR>dog-out</FOR>
	<GIVEN>bowel-problem</GIVEN>
	<GIVEN>family-out</GIVEN>
	<TABLE>0.99 0.01 0.97 0.03 0.9 0.1 0.3 0.7 </TABLE>
</DEFINITION>
```
Then the order of the variables in the table (from left to right) is: "bowel-problem", "family-out", "dog-out". Filling 
in the values, this leads to the following table:

| bowel-problem | family-out    | dog-out   | p    |
|---------------|---------------|-----------|------|
| False         | False         | False     | 0.99 |
| False         | False         | True      | 0.01 |
| False         | True          | False     | 0.97 |
| False         | True          | True      | 0.03 |
| True          | False         | False     | 0.9  |
| True          | False         | True      | 0.1  |
| True          | True          | False     | 0.3  |
| True          | True          | True      | 0.7  |
##### Table 1: CPT of the "dog-out" node of the "dog_problem.BIFXML" example

# The BayesNet Class
Among the files of this project you can find the BayesNet class. We provided you with this class, so you don't need to 
worry much about the data structure in which the BN can be represented. This class provides you with (hopefully) useful
functions for BNs such as loading them from a file and retrieving the CPT of a variable. We *highly* recommend using it.
Of course, you are also free to implement your own methods and change the existing ones if they don't fit your purpose. 
For this class to work you will need to install (either with pip or anaconda) the following packages: networkx, 
matplotlib, pgmpy, pandas (see also requirements.txt).

Internally, the graphical structure of the BN is represented as a DiGraph object from the networkx package. The CPTs
are modelled with DataFrames form the Pandas package. Each CPT is a DataFrame which corresponds to the form as is shown
in Table 1.

We want to point out some methods which we deem especially useful here, but all methods come with their documentation 
in the comments fo the methods.

+ `get_compatible_instantiations_table(instantiation, cpt)`: This method takes an instantiation as a pandas series and
  a CPT as a pandas DataFrame. It checks which rows of the provided CPT are compatible with the instantiation and
  returns only those rows.
  
+ `reduce_factor(instantiation, cpt)`: This method takes an instantiation as a pandas series and
  a CPT as a pandas DataFrame. It returns a new CPT in which all the rows that are incompatible with the instantiation
  are set to 0. 
  
+ `get_interaction_graph()`: Returns a new, undirected Graph object which corresponds to the interaction graph of the 
current BN.
  
+ `draw_structure()`: Plots the graph structure of the current BN.

# Other useful methods
There are also a few other methods which might turn out useful during the implementation of this project. Note that
they are completely optional to use, and it might well be the case that your implementation will work well even without
them. We also provide them as some are used int the BayesNet class.

## Pandas methods
+ `dog_out_CPT['p']` returns the 'p' column of the dog_out_CPT DataFrame as a pandas Series. (0.99 0.01 0.97 0.03 0.9
  0.1 0.3 0.7) 
+ `dog_out_CPT['p'].max()` returns the maximum of the probability values. (0.99)
+ `dog_out_CPT['family-out'] == dog_out_CPT['bowel-problem']` creates a pandas Series which is true for every index in 
  which "bowel-problem" and "family-out" is the same in the dog_out_CPT. (True, True, False, False, False, False, True,
  True). The arguments in the brackets can also be lists of column names in which case multiple columns are compared.
+ `dog_out_CPT.iloc[2]` returns the third row of the DataFrame. (False, True, False, 0.97)
+ `dog_out_CPT.loc[dog_out_CPT['family-out'] == dog_out_CPT['bowel-problem']]` returns all rows of the CPT in which 
    'bowel-problem' == 'family-out'. (rows 0, 1, 6 and 7)
+ `dog_out_CPT.loc[dog_out_CPT['family-out'] == dog_out_CPT['bowel-problem'], 'p'] = 0.0` sets the 'p' value of the
    above mentioned rows to 0.0. This usage of loc[row(s), column(s)] corresponds to direct access to a cell or 
  subtable.
+ `pd.Series({'Winter?': True, 'Sprinkler?': False})` creates a pandas Series in which 'Winter?' is set to True and 
 'Sprinkler?' is set to False. This is a useful format for passing evidence to some methods of this assignment
  
+ `dog_out_CPT.iterrows()` provides an iterator through all the rows of a DataFrame. This is useful for using in loops.
Beware that the returned value is always a tuple of (row_number, row_content).

## Networkx methods
It is likely, that you will not have to use this package at all. One possible methods that could be useful is 
`networkx.neighbors(G, var)` which provides a list of all neighbors of the variable 'var' in the graph 'G'.

