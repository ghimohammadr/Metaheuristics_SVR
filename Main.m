
clc;
clear;
close all;

it = 3;

error = zeros(it,1);
pRMSE = error;
pMAPE = error;
pMAE = error;
c=error;
g=error;
e=error;
bestFitness=error;
times = error;
mse = error;


load('bitc.mat')
load('ybit.mat') 


Crange = [4^-10,4^4]; 
gamma = [4^-10,4^4]; 
epsilon = [4^-10,0.25]; 
pop = 20;
max_iteration = 50;

Data = bitc;
Y = ybit;

for i=1:it
    tic

    
    [row, col] = size(Data);
    PD = 0.80 ;  % percentage 80%
    MainTrainData = Data(1:round(PD*row),:) ; targetData = Y(1:round(PD*row)) ;
    MainTestData = Data(round(PD*row)+1:end,:) ;MainTestTarget = Y(round(PD*row)+1:end) ;
%    [MainTrainIndex, MainTestIndex] = crossvalind('LeaveMOut',row,round((20*row)/100));

    %%% ======= train data
    [row, col] = size(MainTrainData);
    PD2 = 0.80 ;  % percentage 80%
    trainData = MainTrainData(1:round(PD2*row),:) ; trainTarget = targetData(1:round(PD2*row)) ;
    testData = MainTrainData(round(PD2*row)+1:end,:) ;testTarget = targetData(round(PD2*row)+1:end) ;
%    [TrainIndex, TestIndex] = crossvalind('LeaveMOut',row,(20*row)/100); 
    
    [cost,C,G,E,BestFitness] = BOASVMForREG(trainTarget,trainData,testTarget,testData,Crange,gamma,epsilon,pop,max_iteration);

    svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(G),' -p ',num2str(E)];
	model = svmtrain(trainTarget,trainData,svmoptions);
    
    [predict_label, accuracy, prob_estimates] = svmpredict( MainTestTarget, MainTestData,model);

    times(i) = toc    
    c(i)=C
    g(i)=G
    e(i)=E
    bestFitness(i)=BestFitness
    pMAPE(i) = CostFunction(MainTestTarget,predict_label,'MAPE')
    pMAE(i) = CostFunction(MainTestTarget,predict_label,'MAE')
    pRMSE(i) = CostFunction(MainTestTarget,predict_label,'RMSE')
    mse(i) = accuracy(2)
    
    figure; 
    plot(MainTestTarget,'b','LineWidth',2), hold on 
    plot(predict_label,'r','LineWidth',1,'MarkerSize',5) 

    % Observe first hundred points, pan to view more 
    sizeOfTestData = size(MainTestTarget,1);
    xlim([1 sizeOfTestData]) 

    legend({'Actual','Predicted'}) 
    xlabel('Training Data point'); 
    ylabel('Financial Time Series');
    
    
    %%hold on
    figure;
    subplot(2,2,1)       % add first plot in 2 x 2 grid
    plot(times)           % line plot
    title('Times')

    subplot(2,2,2)       % add second plot in 2 x 2 grid
    plot(mse)        % scatter plot
    title('MSE')

    subplot(2,2,4)       % add fourth plot in 2 x 2 grid
    plot(pMAPE)           
    title('MAPE')
    hold off;
    
    drawnow
    
    if it > 1
        clearvars -except mse epsilon bestFitness MainTrainData pop Cvalue max_iteration trainTarget btcdata ybtc trainData testTarget testData targetData MainTestData MainTestTarget gamma Crange pMAPE error c g e bestError Data Y Dataintel Yintel Dataamazon Yamazon Datafacebook Yfacebook Dataapple Yapple instance_matrix label_vector it times bestMSE bestMAPE
    end
%    save(['Data\m',num2str(i)])
end