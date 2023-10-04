%
% Crow Search Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = CSASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];
AP=0.1; % Awareness probability
fl=2; % Flight length (fl)

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
Gbest_position = X(gbestfitindex,:); %globalbest position
Pbest_position = X; %personal best position

for t=1:max_iter
    Iteration = t

    num=ceil(pop*rand(1,pop)); % Generation of random candidate crows for following (chasing)
    for i=1:pop
        if rand>AP
            Xnew(i,:)= X(i,:)+fl*rand*(Pbest_position(num(i),:)-X(i,:)); % Generation of a new position for crow i (state 1)
        else 
            Xnew(i,:)=Lb-(Lb-Ub)*rand; % Generation of a new position for crow i (state 2)
        end
        
        % New Position
        Xnew(i,:) = max(Xnew(i,:), Lb);
        Xnew(i,:) = min(Xnew(i,:), Ub);
    
        %evaluation of fitness function 
        C = Xnew(i,1);
        gam = Xnew(i,2);
        epsil = Xnew(i,3);
    
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
        newfitness(i,1) = accuracy(2); 
        
    end
    
    X = [X
           Xnew];
    fitness = [fitness
                       newfitness];
    [fitness, SortOrder]=sort(fitness);
    Pbest_position = X;
    
    for i=1:pop
        X(i,:)=Pbest_position(SortOrder(i),:);
    end
    X = X(1:pop,:);
    fitness = fitness(1:pop);
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