%
% Artificial Bee Colony Optimization - SVR
%
function [cost,c,g,e,bestFitness] = ABCSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
nOnlooker=pop;         % Number of Onlooker Bees
L=round(0.6*part_dim*pop); % Abandonment Limit Parameter (Trial Limit)
a=1;                    % Acceleration Coefficient Upper Bound
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];


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

[Gbest_fit, gbestfitindex] = min(fitness); %global fitness and its index value
Gbest_position = X(gbestfitindex,:); % globalbest position

% Abandonment Counter
CC=zeros(pop,1);

for t=1:max_iter
    Iteration = t
    % Recruited Bees
    for i=1:pop
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:pop];
        k=K(randi([1 numel(K)]));

        % Define Acceleration Coeff.
        phi=a*unifrnd(-1,+1,[1 part_dim]);

        % New Bee Position
        newbee=X(i,:)+phi.*(X(i,:)-X(k,:));
        newbee= max(newbee, Lb);
        newbee= min(newbee, Ub);
        
        %evaluation of fitness function 
        C = newbee(1);
        gam = newbee(2);
        epsil = newbee(3);
                
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);
    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
                
         MSEtemp = accuracy(2);
         if MSEtemp <fitness(i,1)
              fitness(i,1) = accuracy(2);
              X(i, :) = newbee;
         else
              CC(i)=CC(i)+1;
         end
    end
    
    % Calculate Fitness Values and Selection Probabilities
    F=zeros(pop,1);
    MeanCost = mean(fitness);
    for i=1:pop
        F(i) = exp(-fitness(i)/MeanCost); % Convert Cost to Fitness
    end
    P=F/sum(F);
    
    % Onlooker Bees
    for m=1:nOnlooker
        
        % Select Source Site
        r=rand;
        T=cumsum(P);
        i=find(r<=T,1,'first');
        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:pop];
        k=K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi=a*unifrnd(-1,+1,[1 part_dim]);
        
        % New Bee Position
        newbee=X(i,:)+phi.*(X(i,:)-X(k,:));
        newbee= max(newbee, Lb);
        newbee= min(newbee, Ub);
        
        %evaluation of fitness function
        C = newbee(1);
        gam = newbee(2);
        epsil = newbee(3);
                
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);
    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
                
         MSEtemp = accuracy(2);
         if MSEtemp <fitness(i,1)
              fitness(i,1) = accuracy(2);
              X(i, :) = newbee;
         else
              CC(i)=CC(i)+1;
         end
    end
    
    % Scout Bees
    for i=1:pop
        if CC(i)>=L
            X(i,1) = Crange(1)+(Crange(2)-Crange(1)) * rand(1);
            X(i,2) = gamma(1)+(gamma(2)-gamma(1)) * rand(1);
            X(i,3) = epsilon(1)+(epsilon(2)-Crange(1)) * rand(1);
            C = X(i,1);
            gam = X(i,2);
            epsil = X(i,3);
                
            svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
            model = svmtrain(Ytrain,Xtrain,svmoptions);
    
            [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
            fitness(i) =  accuracy(2);
            
            CC(i)=0;
        end
    end
    
    %update fitness
    [current_gbest_fit,current_index] = min(fitness);
    %update global best fitness and position
    if current_gbest_fit <= Gbest_fit
       Gbest_fit = current_gbest_fit;
       Gbest_position = X(current_index,:);
    end
        
end
        
c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save        
             
end