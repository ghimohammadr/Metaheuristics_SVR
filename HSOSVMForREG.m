%
% Harmony Search Optimization - SVR
%
function [cost,c,g,e,bestFitness] = HSOSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];
nNew=pop - 2;        % Number of New Harmonies
HMCR=0.9;       % Harmony Memory Consideration Rate
PAR=0.1;        % Pitch Adjustment Rate

FW=0.02*(max(Ub)-min(Lb));    % Fret Width (Bandwidth)
FW_damp=0.995;              % Fret Width Damp Ratio


%preallocation of position and fitness
X = zeros(pop,part_dim); %pre_allocation of X position
fitness = zeros(pop,1); %pre_allocation of global fitness function (MSE) value

%%%%% Initialize position and evaluate initial fitness

for i=1:pop
    X(i,1) = Crange(1)+(Crange(2)-Crange(1)) * rand(1);
    X(i,2) = gamma(1)+(gamma(2)-gamma(1)) * rand(1);
    X(i,3) = epsilon(1)+(epsilon(2)-Crange(1)) * rand(1);
    
    C = X(i,1);
    gam = X(i,2);
    epsil = X(i,3);
    
    svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
    model = svmtrain(Ytrain,Xtrain,svmoptions);    
    [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
    fitness(i,1) = accuracy(2);    
end


% Sort
[fitness, SortOrder]=sort(fitness);
Pbest_position = X;
for i=1:pop
    X(i,:)=Pbest_position(SortOrder(i),:);
end


Gbest_fit = fitness(1); %global fitness and its index value
Gbest_position = X(1,:); %globalbest position

for t=1:max_iter
    Iteration = t
    
    
    % Initialize Array for New Harmonies
     newX = zeros(nNew,part_dim);
     newfitness = zeros(nNew,1);
    % Create New Harmonies
    for k=1:nNew

        % Create New Harmony Position
        newX(k,1) = Crange(1)+(Crange(2)-Crange(1)) * rand(1);
        newX(k,2) = gamma(1)+(gamma(2)-gamma(1)) * rand(1);
        newX(k,3) = epsilon(1)+(epsilon(2)-Crange(1)) * rand(1);

        for j=1:part_dim
            if rand<=HMCR
                % Use Harmony Memory
                i=randi([1 pop]);
                newX(k,j) = X(i,j);
            end
            
            % Pitch Adjustment
            if rand<=PAR
                %DELTA=FW*unifrnd(-1,+1);    % Uniform
                DELTA=FW*randn();            % Gaussian (Normal) 
                newX(k,j) = newX(k,j)+DELTA;
            end

        end
        
        % Apply Variable Limits
        newX(k,:)=max(newX(k,:),Lb);
        newX(k,:)=min(newX(k,:),Ub);

        % Evaluation
        C = newX(k,1);
        gam = newX(k, 2);
        epsil = newX(k, 3);
                
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);

        newfitness(k) = accuracy(2);
        
    end

    % Merge Harmony Memory and New Harmonies
    X=[X
        newX];
    fitness = [fitness; newfitness];

    % Sort Harmony Memory
    [fitness, SortOrder]=sort(fitness);
    Pbest_position = X;
    for i=1:pop+nNew
        X(i,:)=Pbest_position(SortOrder(i),:);
    end
    
    % Truncate Extra Harmonies
    X=X(1:pop,:);
    fitness = fitness(1:pop);

    %update fitness
    current_gbest_fit = fitness(1);
    %update global best fitness and position
    if current_gbest_fit <= Gbest_fit
       Gbest_fit = current_gbest_fit;
       Gbest_position = X(1,:);
    end
        
    % Damp Fret Width
    FW=FW*FW_damp;
        
end

c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save


end

