function [ error ] = MAPE(actual, predected)

error = (actual-predected);
squareError = abs(error./predected);
meanSquareError = mean(squareError);
error = meanSquareError;

end

