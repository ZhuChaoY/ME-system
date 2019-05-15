##A Mixture of  Experts and Hierarchical Mixture of  Experts function

#The sigmoid function.
sigmoid = function(x){
  return(1 / (1 + exp(-x)))  
} 

#A function that divides data into random parts.
folds = function(range, N_folds){
  N = length(range)
  folds = list()
  N_per_folds = floor(N / N_folds)
  res = N - N_per_folds * N_folds
  if(res == 0){
    for(i in 1:N_folds){
      folds[[i]] = sample(range, N_per_folds)
      range = setdiff(range, folds[[i]])
    }
  }else{
    for(i in 1:res){
      folds[[i]] = sample(range, N_per_folds + 1)
      range = setdiff(range, folds[[i]])
    }
    for(i in (res + 1):N_folds){
      folds[[i]] = sample(range, N_per_folds)
      range = setdiff(range, folds[[i]])
    }
  }
  return(folds)
}

#A function calculate the area under the curve (AUC).
calculate_auc = function(pre,label){
  order = order(pre, decreasing = T)
  label = label[order]
  N_positive = sum(label == 1)
  N_negative = sum(label == 0)
  auc = (sum(length(pre) + 1 - which(label == 1)) - N_positive * (N_positive + 1) / 2) / N_positive / N_negative
  return(auc)
}

#A function that generates initial parameters.
create_para = function(N_LR, N_SNN, N_input, N_hidden, HME, zero){
  N_EXPERT = N_LR + N_SNN
  W = 2 * matrix(runif(N_LR * (N_input + 1)) - 0.5, N_LR, N_input + 1)
  W1 = 2 * matrix(runif(N_hidden * N_SNN * (N_input + 1)) - 0.5, N_hidden * N_SNN, N_input + 1)
  W2 = 2 * matrix(runif(N_SNN * N_hidden) - 0.5, N_SNN, N_hidden)
  if(!zero){
    W[, N_input + 1] = 0
    W1[, N_input + 1] = 0
  }else{
    W = W - W
    W1 = W1 - W1
    W2 = W2 - W2
  }
  
  if(!HME){
    if(N_LR < 1 || N_SNN < 1){
      stop('Error: Number of Logistic Regression and Single-layer Neural Network for ME system must more than 0')
    }
    v = 2 * matrix(runif(N_EXPERT * N_input) - 0.5, N_EXPERT, N_input)
    if(zero){
      v = v - v
    }
    return(list(v=v, W=W, W1=W1, W2=W2))
  }else{
    if(N_LR < 2 || N_SNN < 2){
      stop('Error: Number of Logistic Regression and Single-layer Neural Network for HME system must more than 1')
    }
    v = 2 * matrix(runif(2 * N_input) - 0.5, 2, N_input)
    v1 = 2 * matrix(runif(N_LR * N_input) - 0.5, N_LR, N_input)
    v2 = 2 * matrix(runif(N_SNN * N_input) - 0.5, N_SNN, N_input)
    if(zero){
      v = v - v
      v1 = v1 - v1
      v2 = v2 - v2
    }
    return(list(v=v, v1=v1, v2=v2, W=W, W1=W1, W2=W2))
  }
}

#A function that calculates the desired indexs based on the current parameters.
calculate_condition = function(para, use_data, use_label, HME, distribution, alpha, statistics, validation){
  N_batch = ncol(use_data)
  N_LR = nrow(para$W)
  N_SNN = nrow(para$W2)
  N_EXPERT = N_LR + N_SNN
  N_hidden = ncol(para$W2)
  brige = rep(N_EXPERT, N_batch)
  
  uu = matrix(0, N_EXPERT, N_batch)
  uu[1:N_LR,] = sigmoid(para$W %*% rbind(use_data, rep(1, N_batch)))    
  for(j in 1:N_SNN){
    uu[N_LR + j, ]=sigmoid(para$W2[j, ] %*% sigmoid(para$W1[(1:N_hidden) + (j - 1) * N_hidden, ] 
                                                   %*% rbind(use_data, rep(1, N_batch))))
  }
  
  if(distribution == 'Gaussian'){
    pp = 1 / sqrt(2 * pi) * exp(-0.5 * (rep(use_label, brige) - uu) ^ 2) + 1e-8
  }else{
    pp=uu ^ rep(use_label, brige) * (1 - uu) ^ (1 - rep(use_label, brige)) + 1e-8
  }
  
  if(!HME){
    g = exp(para$v %*% use_data)
    gg = g / rep(colSums(g), brige)
    h = gg * pp
    hh = h / rep(colSums(h), brige)
  }else{
    g1 = exp(para$v1 %*% use_data)
    gg1 = g1 / rep(colSums(g1), rep(N_LR, N_batch))
    g2 = exp(para$v2 %*% use_data)
    gg2 = g2 / rep(colSums(g2), rep(N_SNN, N_batch))
    g = exp(para$v %*% use_data)
    gg = rbind(gg1, gg2, g / rep(colSums(g), rep(2, N_batch)))
    h = rbind(matrix(rep(gg[N_EXPERT + 1, ], rep(N_LR, N_batch)), N_LR, N_batch), 
              matrix(rep(gg[N_EXPERT + 2, ], rep(N_SNN,N_batch)), N_SNN, N_batch)) * gg[1:N_EXPERT, ] * pp
    H = h / rep(colSums(h), brige)
    hh = rbind(H, colSums(H[1:N_LR, ]), colSums(H[(N_LR + 1):N_EXPERT, ]))
  }
  
  if(statistics){
    if(!HME){
      pre = colSums(gg * uu)
      loss = -sum(hh * log(h)) / N_batch + ifelse(validation, 0, alpha * (sum(diag(para$W %*% t(para$W))) + 
                                                                            sum(diag(para$W2 %*% t(para$W2))) + sum(diag(para$W1 %*% t(para$W1)))))
    }else{
      pre = colSums(rbind(matrix(rep(gg[N_EXPERT + 1, ], rep(N_LR, N_data)), N_LR, N_data),
                          matrix(rep(gg[N_EXPERT + 2, ], rep(N_SNN, N_data)), N_SNN, N_data)) * gg[1:N_EXPERT, ] * uu)
      loss = -sum(H * log(h)) / N_batch + ifelse(validation, 0, alpha * (sum(diag(para$W %*% t(para$W))) + 
                                                                           sum(diag(para$W2 %*% t(para$W2))) + sum(diag(para$W1 %*% t(para$W1)))))
    }
    auc = calculate_auc(pre, use_label)
    return(list(loss=loss, pre=pre, auc=auc))
  }
  else{
    return(list(uu=uu, gg=gg, hh=hh))
  }
}

#A function that update the parameters based on current indexs. 
update_para = function(para, cumulative, momentum, condition, use_data, use_label, HME, distribution, learn_rate, attenuation_rate, momentum_rate, alpha){
  N_batch = ncol(use_data)
  N_LR = nrow(para$W)
  N_SNN = nrow(para$W2)
  N_EXPERT = N_LR + N_SNN
  N_hidden = ncol(para$W2)
  N_input = ncol(para$v)
  uu = condition$uu
  gg = condition$gg
  hh = condition$hh
  
  if(!HME){
    for(j in 1:N_EXPERT){
      g_v = colMeans(diag(gg[j, ] - hh[j, ]) %*% t(use_data))
      cumulative$v[j, ] = attenuation_rate * cumulative$v[j, ] + (1 - attenuation_rate) * g_v ^ 2
      momentum$v[j, ] = momentum_rate * momentum$v[j, ] - learn_rate / sqrt(cumulative$v[j, ] + 1e-8) * g_v
      para$v[j, ] = para$v[j, ] + momentum$v[j, ]
    }
  }else{
    for(j in 1:2){
      g_v = colMeans(diag((gg[N_EXPERT + j, ] - hh[N_EXPERT + j, ])) %*% t(use_data))
      cumulative$v[j, ] = attenuation_rate * cumulative$v[j, ] + (1 - attenuation_rate) * g_v ^ 2
      momentum$v[j, ] = momentum_rate * momentum$v[j, ] - learn_rate / sqrt(cumulative$v[j, ] + 1e-8) * g_v
      para$v[j, ] = para$v[j, ] + momentum$v[j, ]
    }
    
    for(j in 1:N_LR){
      g_v1 = colMeans(diag(hh[N_EXPERT + 1, ] * gg[j, ] - hh[j, ]) %*% t(use_data))
      cumulative$v1[j, ] = attenuation_rate * cumulative$v1[j, ] + (1 - attenuation_rate) * g_v1 ^ 2
      momentum$v1[j, ] = momentum_rate * momentum$v1[j, ] - learn_rate / sqrt(cumulative$v1[j, ] + 1e-8) * g_v1
      para$v1[j, ] = para$v1[j, ] + momentum$v1[j, ]
    }
    
    for(j in 1:N_SNN){
      g_v2 = colMeans(diag(hh[N_EXPERT + 2, ] * gg[j + N_LR, ] - hh[j + N_LR, ]) %*% t(use_data))
      cumulative$v2[j,] = attenuation_rate * cumulative$v2[j, ] + (1 - attenuation_rate) * g_v2 ^ 2
      momentum$v2[j, ] = momentum_rate * momentum$v2[j, ] - learn_rate / sqrt(cumulative$v2[j, ] + 1e-8) * g_v2
      para$v2[j, ] = para$v2[j, ] + momentum$v2[j, ]
    }
  }
  
  for(j in 1:N_LR){
    if(distribution == 'Gaussian'){
      g_W = (colSums(diag((uu[j, ] - use_label) * uu[j, ] * (1 - uu[j, ]) * hh[j, ]) %*%
                       t(rbind(use_data, rep(1, N_batch))))) / N_batch + 2 * alpha * para$W[j, ]
    }else{
      g_W = (colSums(diag((uu[j, ] - use_label) * hh[j, ]) %*% 
                       t(rbind(use_data, rep(1, N_batch))))) / N_batch + 2 * alpha * para$W[j, ]
    }
    cumulative$W[j, ] = attenuation_rate * cumulative$W[j, ] + (1 - attenuation_rate) * g_W ^ 2
    momentum$W[j, ] = momentum_rate * momentum$W[j, ] - learn_rate / sqrt(cumulative$W[j, ] + 1e-8) * g_W
    para$W[j, ] = para$W[j, ] + momentum$W[j, ]
  } 
  
  for(j in 1:N_SNN){
    g_W1 = matrix(0, N_hidden, N_input + 1)
    g_W2 = rep(0, N_hidden)
    inter_para = sigmoid(para$W1[(1:N_hidden) + (j - 1) * N_hidden, ] %*% rbind(use_data, rep(1, N_batch)))
    if(distribution == 'Gaussian'){
      for(i in 1:N_batch){
        inter_matrix = diag(as.vector(inter_para[, i] * (1 - inter_para[, i])))
        g_W1 = g_W1 + (hh[j + N_LR, i] * (uu[j + N_LR, i] - use_label[i]) * uu[j + N_LR, i] * (1 - uu[j + N_LR, i]) * inter_matrix %*%
                         matrix(para$W2[j, ], N_hidden, 1) %*% matrix(c(use_data[, i], 1), 1, N_input + 1)) / N_batch + 
                         alpha * para$W1[(1:N_hidden) + (j - 1) * N_hidden, ]
        g_W2 = g_W2 + (hh[j + N_LR, i] * (uu[j + N_LR, i] - use_label[i]) * uu[j + N_LR, i] * (1 - uu[j + N_LR, i]) * t(inter_para[, i])) / N_batch + 
                       2 * alpha * para$W2[j, ]
      }
    }else{
      for(i in 1:N_batch){
        inter_matrix = diag(as.vector(inter_para[, i] * (1 - inter_para[, i])))
        g_W1 = g_W1 + (hh[j + N_LR, i] * (uu[j + N_LR, i] - use_label[i]) * inter_matrix %*% matrix(para$W2[j, ], N_hidden, 1) %*%
                         matrix(c(use_data[, i], 1), 1, N_input + 1)) / N_batch + alpha * para$W1[(1:N_hidden) + (j - 1) * N_hidden, ]
        g_W2 = g_W2 + (hh[j + N_LR, i] * (uu[j + N_LR, i] - use_label[i]) * t(inter_para[, i])) / N_batch + 2 * alpha * para$W2[j, ]
      }
    } 
    cumulative$W1[(1:N_hidden) + (j - 1) * N_hidden, ] = attenuation_rate * cumulative$W1[(1:N_hidden) +
                                                        (j - 1) * N_hidden, ] + (1 - attenuation_rate) * g_W1 ^ 2
    momentum$W1[(1:N_hidden) + (j - 1) * N_hidden, ] = momentum_rate * momentum$W1[(1:N_hidden) + (j - 1) * N_hidden, ] - learn_rate / 
      sqrt(cumulative$W1[(1:N_hidden) + (j - 1) * N_hidden, ] + 1e-8) * g_W1
    para$W1[(1:N_hidden) + (j - 1) * N_hidden, ] = para$W1[(1:N_hidden) + (j - 1) * N_hidden, ] + momentum$W1[(1:N_hidden) + (j - 1) * N_hidden, ]
    cumulative$W2[j, ] = attenuation_rate * cumulative$W2[j, ] + (1 - attenuation_rate) * g_W2 ^ 2
    momentum$W2[j, ] = momentum_rate * momentum$W2[j, ] - learn_rate / sqrt(cumulative$W2[j, ] + 1e-8) * g_W2
    para$W2[j, ] = para$W2[j, ] + momentum$W2[j, ]
  }
  return(list(para=para, cumulative=cumulative, momentum=momentum))
}

#The main ME function.
#-data	                 The input data as data.frame or matrix.
#-label	                 The input label as row vector. 
#-HME	                   FALSE means a ME system (Default); TRUE means a HME system. 
#-N_LR	                The number of logistic regression model (Default: 2).
#-N_SNN                 The number of Single-layer Neural Network (Default: 2).
#-distribution          Gaussian (Default) or Bernoulli. 
#-N_LR	                The number of hidden layer nodes of Single-layer Neural Network (Default: One less than the number of features).
#-batch_size           The batch size of Stochastic Gradient Descent (Default: 16).
#-epochs          	    The total epoch number (Default: 100). 
#-learn_rate            The learning rate of the Root Mean Square Prop (RMSProp) algorithm (Default: 1e-3). 
#-attenuation_rate     The attenuation rate of learning rate (Default: 0.9).
#-momentum_rate        The momentum rate of the Root Mean Square Prop (RMSProp) algorithm (Default: 0.9).
#-alpha	               The penalty coefficient (Default: 1e-6).
#-print	               TRUE means show the process of convergence (Default); FALSE means hidden convergence process.
#-data_val            Data that is validated simultaneously at iteration time, NULL means no validation data (Default).
#-label_val           Label that is validated simultaneously at iteration time, NULL means no validation label (Default).
ME = function(data, label, HME=FALSE, N_LR=2, N_SNN=2, distribution='Gaussian', N_hidden=ncol(data)-1, batch_size=16,
              epochs=100, learn_rate=1e-3, attenuation_rate=0.9, momentum_rate=0.9, alpha=1e-6, print=TRUE, data_val=NULL, label_val=NULL){
  data = t(data)
  N_data = ncol(data)
  N_input = nrow(data)
  N_EXPERT = N_LR + N_SNN
  iter_per_epoch = floor(N_data / batch_size)
  loss = rep(0, epochs)
  auc = rep(0, epochs)
  
  para = create_para(N_LR, N_SNN, N_input, N_hidden, HME, FALSE)
  cumulative = create_para(N_LR, N_SNN, N_input, N_hidden, HME, TRUE)
  momentum = create_para(N_LR, N_SNN, N_input, N_hidden, HME, TRUE)
  print(paste(ifelse(HME, 'HME', 'ME'), 'system with', N_LR, 'Logistic Regression and,', N_SNN, 'Single-layer Neural Network.'))
  
  if(!is.null(data_val)){
    data_val = t(data_val)
    N_valdata = ncol(data_val)
    loss_val = rep(0, epochs)
    auc_val = rep(0, epochs)
    print(paste('Train on', N_data, 'samples, validate on', N_valdata, 'samples'))
  }else {
    print(paste('Train on', N_data, 'samples'))
  }
  
  for(epoch in 1:epochs){
    if(print){
      print(paste('Epoch ', epoch, '/', epochs))
    }
    samples = folds(1:N_data, iter_per_epoch)
    for(iter in 1:iter_per_epoch){
      sample = samples[[iter]]
      use_data = data[, sample]
      use_label = label[sample]
      N_batch = length(sample)
      
      condition = calculate_condition(para, use_data, use_label, HME, distribution, alpha, FALSE, FALSE)
      newpara = update_para(para, cumulative, momentum, condition, use_data, use_label, HME, distribution, learn_rate, attenuation_rate, momentum_rate, alpha)
      para = newpara$para
      cumulative = newpara$cumulative
      momentum = newpara$momentum
      }
    condition = calculate_condition(para, data, label, HME, distribution, alpha, TRUE, FALSE)
    pre = condition$pre
    loss[epoch] = condition$loss
    auc[epoch] = condition$auc
    
    if(!is.null(data_val)){
      condition_val = calculate_condition(para, data_val, label_val, HME, distribution, alpha, TRUE, TRUE)
      pre_val = condition_val$pre
      loss_val[epoch] = condition_val$loss
      auc_val[epoch] = condition_val$auc
    }
    
    if(print){
      if(HME){
        print(paste('loss: ', loss[epoch], '; auc: ', auc[epoch], '; loss_val: ', loss_val[epoch], '; auc_val: ', auc_val[epoch]))
      }else{
        print(paste('loss: ', loss[epoch], '; auc: ', auc[epoch]))
      }
    }
  }

  if(!is.null(data_val)){
    return(list(para=para, pre=pre, pre_val=pre_val, loss=loss, auc=auc, loss_val=loss_val, auc_val=auc_val))
  }else{
    return(list(para=para, pre=pre, loss=loss, auc=auc))
  }
}
   

