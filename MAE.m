function [ error ] = MAE(actual, predected)

error = (actual-predected);
squareError = abs(error);
meanSquareError = mean(squareError);
error = meanSquareError;

end

