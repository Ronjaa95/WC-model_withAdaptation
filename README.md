# WC-model_withAdaptation
In this framework, the neurolib-repository by https://github.com/neurolib-dev/neurolib (version 0.5.13) is expanded by a Wilson-Cowan model with a hyperpolarizing adaptation mechanism, generating slow oscillations in a connectome.

The additional provided notebooks are visualizations and computations of the data acqusition on slow oscillations. The python-sripts are the derivation functions, to gather certain aspects of the model. They provide nullcline-, and fixed point derivations for the single node model (linear stability analysis), as well as a bistability analysis on the whole-brain-network (numerical stability analysis) for which another model in the neurolib/neurolib/models/ can be found, named 'wc_input'. It adds a step current at two time points (choosable). The DataFrame computation functions are only templates, on how to initialize such a derivation.

Documentation of theoretical background and output/results of the system in a master's thesis: 'The Wilson-Cowan model with adaptation as a simplified model for slow oscialltions', at the Technische Universität Berlin.
