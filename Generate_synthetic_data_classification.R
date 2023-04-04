############################################################################
# Generate simulated data using gLV model for COP of classification purpose
#Last update: 2023-03-04, Xu-Wen Wang
############################################################################
library("seqtime")

setwd("/udd/spxuw/COP/code")

N =100
M = 1100

set.seed(123)
desired_count = c(550,550)

# different network connectivity
for (N_sub in c(30)){
  for (C in c(0.4)){
    success_flag = rep(0,length(desired_count))
    set.seed(234)
    A = (matrix(rnorm(N*N,mean=0,sd=0.1), N, N))
    set.seed(234)
    Connect = sample(0:1, N^2, prob = c(1-C, C), replace = TRUE)
    A = A * matrix(Connect, N, N)
    diag(A) = -1
    set.seed(123)
    b = runif(N)
  
    final_relative = matrix(0,N,M)
    final_absolute = matrix(0,N,M)
    initial_relative = matrix(0,N,M)
    initial_absolute = matrix(0,N,M)
    
    initial_seed = 1
    while (sum(success_flag)<M) {
      print(sum(success_flag))
      # before adding taxa 1
      set.seed(initial_seed)
      collection = sample(2:N,N_sub)
      y_0 = runif(N)
      y_0[setdiff(1:N,collection)] = 0
      x = glv(N = N, A, b = b, y = y_0, tstart = 0, tend = 100, tstep = 0.1, perturb = NULL)
      # after adding taxa 1
      y_1 = x[,ncol(x)]
      y_1[y_1<0] = 0
      y_1[1] = 0.2
      x = glv(N = N, A, b = b, y = y_1, tstart = 0, tend = 100, tstep = 0.1, perturb = NULL)
      x[x<0] = 0
      # check the abundance
      if ((x[1,ncol(x)]>0.05)&&(success_flag[2]<desired_count[2])){
        success_flag[2] = success_flag[2]+1
        initial_relative[,sum(success_flag)] = y_1/sum(y_1)
        final_relative[,sum(success_flag)] = x[,ncol(x)]/sum(x[,ncol(x)])
        final_absolute[,sum(success_flag)] = x[,ncol(x)]
        initial_absolute[,sum(success_flag)] = y_1
      }
    if ((x[1,ncol(x)]<=0.05)&&(success_flag[1]<desired_count[1])){
      success_flag[1] = success_flag[1]+1
      initial_relative[,sum(success_flag)] = y_1/sum(y_1)
      final_relative[,sum(success_flag)] = x[,ncol(x)]/sum(x[,ncol(x)])
      final_absolute[,sum(success_flag)] = x[,ncol(x)]
      initial_absolute[,sum(success_flag)] = y_1
    }
      initial_seed = initial_seed + 1
    }
  
    for (f in c(0.5,1,2,5,10)){
      for (fold in 1:5){
        set.seed(fold)
        test_index = sample(1:M,100)
        set.seed(fold)
        train_index = sample(setdiff(1:M,test_index),f*N)
        Ztrain_relative = initial_relative[,train_index]
        Ztest_relative = initial_relative[,test_index]

        Ztrain_absolute = initial_absolute[,train_index]
        Ztest_absolute = initial_absolute[,test_index]      

        label_train_relative = as.numeric(final_relative[1,train_index])
        label_test_relative = as.numeric(final_relative[1,test_index])

        label_train_absolute = as.numeric(final_absolute[1,train_index])
        label_test_absolute = as.numeric(final_absolute[1,test_index])
    
        write.table(Ztrain_relative, file = paste('../data/gLV/sigma_0.1/classification/gLV_sub_',N_sub,'_C_',C,'_X_train_relative_',f,'_',fold,'.csv',sep=''), row.names = F, col.names = F, sep=",")
        write.table(Ztest_relative, file = paste('../data/gLV/sigma_0.1/classification/gLV_sub_',N_sub,'_C_',C,'_X_test_relative_',f,'_',fold,'.csv',sep=''), row.names = F, col.names = F, sep=",")
        write.table(Ztrain_absolute, file = paste('../data/gLV/sigma_0.1/classification/gLV_sub_',N_sub,'_C_',C,'_X_train_absolute_',f,'_',fold,'.csv',sep=''), row.names = F, col.names = F, sep=",")
        write.table(Ztest_absolute, file = paste('../data/gLV/sigma_0.1/classification/gLV_sub_',N_sub,'_C_',C,'_X_test_absolute_',f,'_',fold,'.csv',sep=''), row.names = F, col.names = F, sep=",")
        write.table(label_train_relative, file = paste('../data/gLV/sigma_0.1/classification/gLV_sub_',N_sub,'_C_',C,'_y_train_relative_',f,'_',fold,'.csv',sep=''), row.names = F, col.names = F, sep=",")
        write.table(label_test_relative, file = paste('../data/gLV/sigma_0.1/classification/gLV_sub_',N_sub,'_C_',C,'_y_test_relative_',f,'_',fold,'.csv',sep=''), row.names = F, col.names = F, sep=",")
        write.table(label_train_absolute, file = paste('../data/gLV/sigma_0.1/classification/gLV_sub_',N_sub,'_C_',C,'_y_train_absolute_',f,'_',fold,'.csv',sep=''), row.names = F, col.names = F, sep=",")
        write.table(label_test_absolute, file = paste('../data/gLV/sigma_0.1/classification/gLV_sub_',N_sub,'_C_',C,'_y_test_absolute_',f,'_',fold,'.csv',sep=''), row.names = F, col.names = F, sep=",")
      }
    }
  }
}

