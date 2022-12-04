The data and results in the repository are complete as reported in the research "Machine Learning Augmented Approaches for Hub Location Problems". The paper is under review now.

### Data

The folder data in this repository includes synthetic networks used in our experiments. All synthetic networks are created according to the method described in the paper. The folder data consists of three folders as:

- SYN25：
  - training set: 10,000 synthetic networks with 25 nodes, generated with different random seeds.
  - validation set: 500 synthetic networks with 25 nodes, generated with different random seeds.
  - test set: 500 synthetic networks with 25 nodes,  generated with different random seeds.
- SYN100: 10 synthetic networks with 100 nodes, which are used to test DL-GVNS.
- SYN200: 10 synthetic networks with 200 nodes, which are used to test DL-GVNS.

In each synthetic dataset like SYN100, there are four files named :

`x_list.pickle`: a list contains all lists of coordinate $x$ of each node in all networks, like $[[x_1^{G_1}, x_2^{G_1}, .., x_N^{G_1}], [x_1^{G_2}, x_2^{G_2}, .., x_N^{G_2}],...]$.

`y_list.pickle` : a list contains all lists of coordinate $y$ of each node in all networks, like $[[y_1^{G_1}, y_2^{G_1}, .., y_N^{G_1}], [y_1^{G_2}, y_2^{G_2}, .., y_N^{G_2}],...]$.

`distance_list.pickle`: a list contains distance matrix of each network.

`demand_list.pickle`: a list contains demand matrix of each network.

The `.pickle` can be loaded use the following codes, and please refer to  https://docs.python.org/3/library/pickle.html for learning more about `pickle` module.

```python
import pickle as pkl
with open ('x_list.pickle', 'rb') as f:
	x_list = pkl.load(f)
```

In addition to synthetic data, we use standard datasets derived from real-world and 

CAB data is provided by https://www.researchgate.net/project/Studies-in-Hub-Location-and-Network-Design;

TR data can be obtained from 

AP dataset can be fetched on the website: https://users.monash.edu/~andrease/Downloads.htm.

 #### Results

The folder results in this repository includes the exact objective (solution) and  running time of all problems test in the paper, with the proposed and benchmark algorithms.

- DL-CBS
  - (DL)CBS_results.csv:  solutions and running time of 224 problems, solved with nine approaches: RCBS, ARCBS, IRCBS, IARCBS, DL-RCBS, DL-ARCBS, DL-IRCBS, DL-IARCBS. The limitation time of AP200 is seven hours while four hours for other datasets.
  - ap_24h_DL_IARCBS.csv: solutions and running time of AP200 , solved with IARCBS, DL-IARCBS and CPLEX. It is noted that the limitation time is 24 hours.
- DL-GVNS
  - (DL)GVNS_synthetic.csv: comparison of DL-GVNS and Mix-GVNS, on 10 synthetic networks with 100 nodes and 10 synthetic networks with 200 nodes.
  - (DL)GVNS_real_world.csv: comparison of DL-GVNS and Mix-GVNS on real-world datasets.
  - compare_with_GA_TS.csv: comparison of DL-GVNS, genetic algorithm (GA), tabu search (TS) and CPLEX on real-word datasets.