%
% Butterfly Optimization Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = BOASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];
p=0.8;                       % probabibility switch
power_exponent=0.1;
sensory_modality=0.01;

%preallocation of butterfly position and fitness
X = zeros(pop,part_dim); %pre_allocation of X butterfly position
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
Pbest_position = X; % personal best position
fMSE = fitness;

for t = 1:max_iter %its ietration number
     Iteration = t
     
     for i=1:pop, % Loop over all butterflies/solutions
         
        %Calculate fragrance of each butterfly which is correlated with objective function
        newfitness = fMSE(i,1);
        
        FP=(sensory_modality*(newfitness^power_exponent));   
    
        %Global or local search
        if rand<p,    
           dis = rand * rand * Gbest_position - X(i,:);        
           Pbest_position(i,:)=X(i,:)+dis*FP;
        else
            % Find random butterflies in the neighbourhood
            epsilon=rand;
            JK=randperm(pop);
            dis=epsilon*epsilon*X(JK(1),:)-X(JK(2),:);
            Pbest_position(i,:)=X(i,:)+dis*FP;                         
        end
                   
        % Check if the simple limits/bounds are OK
        Pbest_position(i,:)= max( Pbest_position(i,:), Lb);
        Pbest_position(i,:)= min(Pbest_position(i,:),Ub);
        % Evaluate new solutions
        C = Pbest_position(i,1);
        gam = Pbest_position(i, 2);
        epsil = Pbest_position(i, 3);
                
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);
    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
        newfitness = accuracy(2);
        
        % If fitness improves (better solutions found), update then
        if (newfitness<fitness(i)),
            X(i,:)=Pbest_position(i,:);
            fitness(i)=newfitness;
        end
        
     end
     
    [current_gbest_fit,current_index] = min(fitness);
    %update global best fitness and position 
    if current_gbest_fit <= Gbest_fit
       Gbest_fit = current_gbest_fit;
       Gbest_position = X(current_index,:);
    end
       
    %Update sensory_modality
    sensory_modality=sensory_modality+(0.025/(sensory_modality*max_iter));
    fMSE = fitness;
end

c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save

end


