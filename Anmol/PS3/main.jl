using QuantEcon

z_chain = MarkovChain([0.9 0.1; 0.1 0.9], [0.1; 1.0]);
z_size = length(z_chain.state_values);