
6th November 2017
-----------------

Trained with SGDR, where at each reset of the LR to the maximum we fix
any weights below 0.02 to zero. This creates a quickly decreasing set of weights.
 Although, we lost a relatively 
large amount of the accuracy;  Was 94.25% before sparsification and is 90.39% 
afterwards. May be able to improve that by lowering the threshold for weight 
removal, or setting it based on the proportion of weights to remove at each
step. Would probably be best to do that with a final limit to the number of 
weights.

For 90.39%, only 0.5286% of the parameters are nonzero.

This is actually in line with the results of training in the "traditional"
deep compression way: after initial convergence set all parameters below
0.02, decrease the learning rate and continue training for a relatively long
time with a fixed learning rate. So, this method with SGDR doesn't appear to
be any worse.
