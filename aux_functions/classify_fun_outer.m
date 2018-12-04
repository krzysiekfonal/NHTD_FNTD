function F = classify_fun_outer(xtrain,ytrain,xtest,algorithm,param)

        [y_test,info_legend] = classify_fun(xtrain,ytrain,xtest,algorithm,param);
        F = {y_test, info_legend};

end