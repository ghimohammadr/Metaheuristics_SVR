%
% Invasive Weed Optimization - SVR
%
function [cost,c,g,e,bestFitness] = IWOSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop0 = population;
pop = population+ 15;     % Initial Population Size
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];
Smin = 0;       % Minimum Number of Seeds
Smax = 5;       % Maximum Number of Seeds
Exponent = 2;           % Variance Reduction Exponent
sigma_initial = 0.5;    % Initial Value of Standard Deviation
sigma_final = 0.001;	% Final Value of Standard Deviation

%preallocation of position and fitness
X = zeros(pop0,part_dim); %pre_allocation of X position
fitness = zeros(pop0,1); %pre_allocation of global fitness function (MSE) value

%%%%% Initialize position and evaluate initial fitness

for i=1:pop0
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

[Gbest_fit, gbestfitindex] = min(fitness); %global fitness and its index value
Gbest_position = X(gbestfitindex,:); %globalbest position

for t=1:max_iter
    Iteration = t
    
    % Update Standard Deviation
    sigma = ((max_iter - t)/(max_iter - 1))^Exponent * (sigma_initial - sigma_final) + sigma_final;
    
    % Get Best and Worst Cost Values
    Costs = fitness;
    BestCost = min(Costs);
    WorstCost = max(Costs);
    
    % Initialize Offsprings Population
    newX = [];
    newfitness = [];
    
    % Reproduction
    for i = 1:pop0
        
        ratio = (fitness(i) - WorstCost)/(BestCost - WorstCost);
        S = floor(Smin + (Smax - Smin)*ratio);
        
        for j = 1:S
                      
            % Generate Random Location
            newsolposition = X(i,:) + sigma*(Lb+(Ub-Lb)).*randn(1, part_dim);
            
            % Apply Lower/Upper Bounds
            newsolposition = max(newsolposition,Lb);
            newsolposition = min(newsolposition,Ub);
            
            % Evaluate Offsring            
            C = newsolposition(1);
            gam = newsolposition(2);
            epsil = newsolposition(3);
    
            svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
            model = svmtrain(Ytrain,Xtrain,svmoptions);   
            [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
      
            newsolcost = accuracy(2);              
            
            % Add Offpsring to the Population
            newX = [newX
                        newsolposition];  %#ok               
            newfitness = [newfitness
                                newsolcost];
        end
        
    end

    % Merge Populations
    X = [X
            newX];  %#ok               
    fitness = [fitness
                        newfitness];

    % Sort
    [fitness, SortOrder]=sort(fitness);
    Pbest_position = X;
    for i=1:size(X,1)
        X(i,:) = Pbest_position(SortOrder(i),:);
    end

    % Competitive Exclusion (Delete Extra Members)
    if size(X,1)> pop
        X = X(1:pop,:);
        fitness = fitness(1:pop);
    end

    %update fitness
    current_gbest_fit = fitness(1);
    %update global best fitness and position 
    if current_gbest_fit <= Gbest_fit
       Gbest_fit = current_gbest_fit;
       Gbest_position = X(1,:);
    end 

end
     
c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save 

end