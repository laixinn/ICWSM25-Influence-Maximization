# Readme for read_tweet.py

### Graph construction

```jsx
Please run the following code in read_tweet.py.
bg = build_tgraph()
'''week accmulative graph'''
bg.week_graph()
'''find strong tie according to graph frequent'''
bg.strong_tie_v2()
```

- week_graph() method constructs the weekly weak graph.

- strong_tie_v2() method constructs the strong graph based on the weak graph. Specifically, it filters by the weight according to the user-defined MEDIAN value.

### Labeling

Please run label.ipynb.

### Sample weak graph

```jsx
Please only run bg.sample_weak_graph() in read_tweet.py.
```

The main steps are:

1. Load Labeled Nodes.

2. Load the Strong Graph, and then find the neighboring nodes of the Labeled Nodes within it.

3. Based on these neighbor nodes, two situations are defined:

    a. If the Labeled Nodes account for more than 22% of the total nodes (including neighbor nodes), then randomly select some of the unlabeled neighboring nodes to increase the total number of nodes, so that the proportion of labeled nodes decreases to around 20%.

    b. If the Labeled Nodes account for less than 18% of the total nodes (including neighbor nodes), then find nodes from the edges that contain the Labeled Nodes, and add them to the node set to increase the proportion of labeled nodes to around 18%.

Finally, if the proportion is already within the 18%-22% range, directly save the sampling result and update the labeled nodes.
