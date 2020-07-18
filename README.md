RecCNTZST
---
Carbon nanotube zero skew clock tree construction using structural recursive neural network.

#### Structure
- config: configuration for the network/etc.
- optimizer: backward propagation.** And the launcher of the program.**
- parse_out: parse the final output from the output of network. 
- parse_topo: parse the tree-topo from the source.
- read_source: read the source from the file.
- recursive: construct the network using the topology.
- util: some tools.


#### Workflow
Figure version interpretation<br>
![Figure](http://a1.qpic.cn/psc?/05f296b7-f920-4499-af25-b1090ac6d0d1/4KbNA3H1osI2VUAtoM9GOo5hdzsV4HwZpUuXromGJaubUbAebqGcBbbQJSTPGBVqCdvl4f818ASrJAN1AVUV0w!!/b&ek=1&kp=1&pt=0&bo=HwSlAh8EpQIRADc!&tl=1&tm=1595077200&sce=0-12-12&rf=viewer_311)

Algorithm version interpretation<br>
![Algorithm](http://a1.qpic.cn/psc?/05f296b7-f920-4499-af25-b1090ac6d0d1/4KbNA3H1osI2VUAtoM9GOk0KKRshm6ixvmkEOnpKIjd1FcVXHRd82*uBUPIBpgadoPksWYHm4T*4yuXQUTt5rw!!/b&ek=1&kp=1&pt=0&bo=hgLzAoYC8wIRADc!&tl=1&vuin=1057398161&tm=1595077200&sce=50-1-1&rf=viewer_311)

Loaded network interpretation<br>
![Loaded network](http://a1.qpic.cn/psc?/05f296b7-f920-4499-af25-b1090ac6d0d1/4KbNA3H1osI2VUAtoM9GOvi*kP8szqSbKEiRp7hPOFJA.0*3zAg0oy.eUT75S9y.353VYcTCmgSEiXnQc2XO9A!!/b&ek=1&kp=1&pt=0&bo=LgZPAy4GTwMRADc!&tl=1&vuin=1057398161&tm=1595077200&sce=50-1-1&rf=viewer_311)

#### Basis
- Carbon Nanotube
```
trainable params: wire length, cnt diameter, bundle diameter
generated params: coordinates, number of bends
```

- Clock Tree Synthesis
```
minimizing constraint: T = maximum delay
equality constraint: E = max_delay - min_delay = 0
loss function: L = T + \lambda * E
```