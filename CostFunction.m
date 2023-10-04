function [ error ] = CostFunction(actual, predected,func)

if strcmp(func,'RMSE')
   error = RMSE(actual,predected);
elseif strcmp(func,'MAPE')
   error = MAPE(actual,predected);
elseif strcmp(func,'MAE')
   error = MAE(actual,predected);
else
    error = immse(actual,predected);
end

end

