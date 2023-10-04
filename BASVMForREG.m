%
% Bees Algorithm Optimization - SVR
%
function [cost,c,g,e,bestFitness] = BASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];                           % Number of Scout Bees
nSelectedSite=round(0.5*pop);     % Number of Selected Sites
nEliteSite=round(0.4*nSelectedSite);    % Number of Selected Elite Sites
nSelectedSiteBee=round(0.5*pop);  % Number of Recruited Bees for Selected Sites
nEliteSiteBee=2*nSelectedSiteBee;       % Number of Recruited Bees for Elite Sites
r=0.1*(max(Ub)-min(Lb));	% Neighborhood Radius
rdamp=0.95;             % Neighborhood Radius Damp Rate


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

for t = 1:max_iter %its ietration number
    Iteration = t
    % Elite Sites
    for i=1:nEliteSite
        
        bestnewbeeCost=inf;
        
        for j=1:nEliteSiteBee  
            y=X(i,:);
            k=randi([1 part_dim]);    
            y(k)=X(i,k)+unifrnd(-r,r);
            newbeePosition=y;
            newbeePosition= max(newbeePosition, Lb);
            newbeePosition= min(newbeePosition, Ub);
            
            %evaluation of fitness function
            C = newbeePosition(1);
            gam = newbeePosition(2);
            epsil = newbeePosition(3);
                
            svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
            model = svmtrain(Ytrain,Xtrain,svmoptions);    
            [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
                
            if accuracy(2) < bestnewbeeCost
                bestnewbeeCost = accuracy(2);
            end
       end
       if bestnewbeeCost <fitness(i,1)
            fitness(i,1) = bestnewbeeCost;
            X(i, :) = newbeePosition;
       end
    end
    
    % Selected Non-Elite Sites
    for i=nEliteSite+1:nSelectedSite
        
        bestnewbeeCost=inf;
        
        for j=1:nSelectedSiteBee
            k=randi([1 part_dim]);    
            y=X(i,:);
            y(k)=X(i,k)+unifrnd(-r,r);
            newbeePosition=y;
            newbeePosition= max(newbeePosition, Lb);
            newbeePosition= min(newbeePosition, Ub);
            
            %evaluation of fitness function 
            C = newbeePosition(1);
            gam = newbeePosition(2);
            epsil = newbeePosition(3);
                
            svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
            model = svmtrain(Ytrain,Xtrain,svmoptions);    
            [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
                
            if accuracy(2) < bestnewbeeCost
                bestnewbeeCost = accuracy(2);
            end
       end
       if bestnewbeeCost <fitness(i,1)
            fitness(i,1) = bestnewbeeCost;
            X(i, :) = newbeePosition;
       end
    end

        % Non-Selected Sites
    for i=nSelectedSite+1:pop
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

    %update fitness
    current_gbest_fit = fitness(1);
    %update global best fitness and position 
    if current_gbest_fit <= Gbest_fit
       Gbest_fit = current_gbest_fit;
       Gbest_position = X(1,:);
    end
    
    % Damp Neighborhood Radius
    r=r*rdamp;

end

c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save

end