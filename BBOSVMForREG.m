%
% Biogeography-Based Optimization - SVR
%
function [cost,c,g,e,bestFitness] = BBOSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];
KeepRate=0.2;                   % Keep Rate
nKeep=round(KeepRate*pop);     % Number of Kept Habitats
nNew=pop-nKeep;                % Number of New Habitats

% Migration Rates
mu=linspace(1,0,pop);          % Emmigration Rates
lambda=1-mu;                    % Immigration Rates
alpha=0.9;
pMutation=0.1;
sigma=0.02*(max(Ub)-min(Lb));


%preallocation of position and fitness
X = zeros(pop,part_dim); %pre_allocation of X firefly position
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
Gbest_position = X(1,:); % globalbest position

for t = 1:max_iter %its ietration number
     Iteration = t

   % Initialize Array for New population
     newX = zeros(pop,part_dim);
     newfitness = zeros(pop,1);
    
     for i=1:pop
        for k=1:part_dim
            % Migration
            if rand<=lambda(i)
                 % Emmigration Probabilities
                 EP=mu;
                 EP(i)=0;
                 EP=EP/sum(EP);

                 % Select Source Habitat
                 r=rand;
                 CC=cumsum(EP);
                 j=find(r<=CC,1,'first');
             
                 % Migration
                 newX(i,k)=newX(i,k)+alpha*(newX(j,k)-newX(i,k));
            end
            
            % Mutation
            if rand<=pMutation
                newX(i,k)=newX(i,k)+sigma*randn;
            end
        end
        % Apply Variable Limits
        newX(i,:)=max(newX(i,:),Lb);
        newX(i,:)=min(newX(i,:),Ub);             
             
        % Evaluation
        C = newX(i,1);
        gam = newX(i, 2);
        epsil = newX(i, 3);
                
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
        
        newfitness(i) = accuracy(2);

     end
          
    % Sort New Population
    [newfitness, SortOrder]=sort(newfitness);
    Pbest_position = newX;
    for i=1:pop
        newX(i,:)=Pbest_position(SortOrder(i),:);
    end

    % Select Next Iteration Population
    X=[X(1:nKeep,:)
         newX(1:nNew,:)];
    fitness =[fitness(1:nKeep); newfitness(1:nNew)];
    
    % Sort Population
    [fitness, SortOrder]=sort(fitness);
    Pbest_position = X;
    for i=1:pop
        X(i,:)=Pbest_position(SortOrder(i),:);
    end
    
    % Update Best Solution Ever Found
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