#A normalize function
normalize = function(x){
  return((x - min(x)) / (max(x) - min(x)))
}  

#Iris dataset 
data('iris')
label = rep(c(0, 1),c(50, 50)) 
data = apply(iris[1:100, -5], 2, normalize)

#A use demonstration
model = ME(data, label, HME=FALSE, N_LR=2, N_SNN=2, distribution='Gaussian', N_hidden=ncol(data)-1, batch_size=16,
           epochs=50, learn_rate=1e-3, attenuation_rate=0.9, momentum_rate=0.9, alpha=1e-6, print=TRUE, data_val=NULL, label_val=NULL)
