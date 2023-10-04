function [ error ] = RMSE(actual, predected)

error = (actual-predected);
squareError = error.^2;
meanSquareError = mean(squareError);
error = sqrt(meanSquareError);

end

