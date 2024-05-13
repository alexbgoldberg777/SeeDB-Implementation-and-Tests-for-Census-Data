# SeeDB-Implementation-and-Tests-for-Census-Data
This project was created for the collaborators' UMass Amherst Computer Science 645 final project.

SeeDB (link: https://dl.acm.org/doi/10.14778/2831360.2831371) is a system that provides recommendations for the most interesting aggregate visualizations of a dataset based on the value deviation of a desired target query and a reference query. This is done using optimization algorithms that find the best aggregate views of the form:

$$SELECT \space a, \space f(m) \space from \space D \space GROUP \space BY \space a$$

where D is the dataset, a is a feature being aggregated by, m is a feature being aggregated, and f is an aggregate function. These are found from the set of all possible combinations of all categorical features A from D, all possible numerical features M, and of the set of most common aggreate functions F={'sum', 'mean', 'max', 'min', 'count'}. In this project, we provide our own implementations of these optimization algorithms (in algorithms.py) and use this implementation to find the top 5 most interesting views and visualizations on the [1994 U.S. Adult Census dataset](https://archive.ics.uci.edu/dataset/20/census+income). With this data, we use the subset of entries of married people as the target query, and the subset of unmarried people as the references query in order to find the most interesting deviations between married and unmarried people.
