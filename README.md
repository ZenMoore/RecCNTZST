RecCNTZST
---
Carbon nanotube zero skew clock tree construction
 using structural recursive neural network.
 
#### Structure
- config: configuration for the network/etc.
- optimizer: backward propagation.**The launcher of the program.**
- parse_out: parse the final output from the output of network. 
- parse_topo: parse the tree-topo from the source.
- read_source: read the source from the file.
- recursive: construct the network using the topology.
- util: some tools.

 
#### Workflow
1. read source file to get the sink set recording the sink-resistances and sink-capacitances/etc.
2. generate the tree topology and return the root to recursive.py
3. construct the RecNet according to the tree-topo.
4. initialize the tree by initial values(or DME).
5. calculate the loss of the whole tree(total delay + lambda operator * skew)
6. gradient descend according to the gotten loss along the operating graph.
7. repeat the backward algorithmuntil reaching the extreme point.(some
techniques for jumping out of local minumum).
8. stop the descending and parse out the results from the states of all nodes in RecNet.

#### Basis
- Carbon Nanotube
```
state params: wire length, doping proportion, diameter
hyper params: buffer sequence, cnt type sequence
```

- Clock Tree Synthesis
```
minimizing constraint: total delay
equality constraint: max delay - min delay = 0
```