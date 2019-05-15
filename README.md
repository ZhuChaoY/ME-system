ME:  A Mixture of  Experts and Hierarchical Mixture of  Experts function

I. Introduction

    ME system decomposes a complex problem into many simple sub-problems, and these 
    sub-problems split the input space into regions where particular local experts can specialize.
    HME system is the multi-level ME system.

II. System requirements

    R (Above 3.5)

III. Files

    ME.R	        The main algorithm script.
    Sample.R	    A use demonstration (iris dataset).

IV. Functions in ME.R
      
     sigmoid( )                      The sigmoid function.
     folds( )                         A function that divides data into random parts.
     calculate_auc( )                 A function calculate the area under the curve (AUC).
     create_para( )                   A function that generates initial parameters.
     calculate_condition( )           A function that calculates the desired indexs based on the current parameters.
     update_para( )                   A function that update the parameters based on current indexs. 
     ME( )                            The main ME function.

V.  Options in ME function:
				
     -data	                 The input data as data.frame or matrix.
     -label	                 The input label as row vector. 
     -HME	                   FALSE means a ME system (Default); TRUE means a HME system. 
     -N_LR	                 The number of logistic regression model (Default: 2).
     -N_SNN                  The number of Single-layer Neural Network (Default: 2).
     -distribution           Gaussian (Default) or Bernoulli. 
     -N_LR	                 The number of hidden layer nodes of Single-layer Neural Network (Default: One less than the number of                                  features).
     -batch_size             The batch size of Stochastic Gradient Descent (Default: 16).
     -epochs             	   The total epoch number (Default: 100). 
     -learn_rate             The learning rate of the Root Mean Square Prop (RMSProp) algorithm (Default: 1e-3). 
     -attenuation_rate       The attenuation rate of learning rate (Default: 0.9).
     -momentum_rate          The momentum rate of the Root Mean Square Prop (RMSProp) algorithm (Default: 0.9).
     -alpha	                 The penalty coefficient (Default: 1e-6).
     -print	                 TRUE means show the process of convergence (Default); FALSE means hidden convergence process.
     -data_val               Data that is validated simultaneously at iteration time, NULL means no validation data (Default).
     -label_val              Label that is validated simultaneously at iteration time, NULL means no validation label (Default).

VI. Contact:

    If you have any questions, please contact: 18111510028@fudan.edu.cn
