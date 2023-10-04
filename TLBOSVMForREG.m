%
% Teaching-Learning-Based Optimization - SVR
%
function [cost,c,g,e,bestFitness] = TLBOSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];

%preallocation of position fitness
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

for t = 1:max_iter %its ietration number
     Iteration = t

    % Calculate Population Mean
    Mean = 0;
    for i=1:pop
        Mean = Mean + X(i,:);
    end
    Mean = Mean/pop;
    
        % Select Teacher
    Teacherposition = X(1,:);
    Teachercost = fitness(1);
    for i=2:pop
        if fitness(i) < Teachercost
            Teacherposition = X(i,:);
            Teachercost = fitness(i);
        end
    end

   % Teacher Phase
    for i=1:pop
        % Create Empty Solution
        newsolposition = zeros(1,part_dim); 
        
        % Teaching Factor
        TF = randi([1 2]);
        
        % Teaching (moving towards teacher)
        newsolposition = X(i,:)+ rand(1, part_dim).*(Teacherposition - TF*Mean);
        
        % Clipping
        newsolposition = max(newsolposition,Lb);
        newsolposition = min(newsolposition,Ub);
        
        % Evaluation        
        C = newsolposition(1);
        gam = newsolposition(2);
        epsil = newsolposition(3);
    
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
        newsolcost = accuracy(2);    
        
        % Comparision
        if newsolcost<fitness(i)
            X(i,:) = newsolposition;
            fitness(i) = newsolcost;
        end
    end

    % Learner Phase
    for i=1:pop
        
        A = 1:pop;
        A(i)=[];
        j = A(randi(pop-1));
        
        Step = X(i,:) - X(j,:);
        if fitness(j) < fitness(i)
            Step = -Step;
        end
        
        % Create Empty Solution
        newsolposition = zeros(1,part_dim);
        newsolcost = zeros(1,1);
        
        % Teaching (moving towards teacher)
        newsolposition = X(i,:)+ rand(1, part_dim).*Step;
        
                % Clipping
        newsolposition = max(newsolposition,Lb);
        newsolposition = min(newsolposition,Ub);
        
        % Evaluation        
        C = newsolposition(1);
        gam = newsolposition(2);
        epsil = newsolposition(3);
    
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
        newsolcost = accuracy(2);    
        
        % Comparision
        if newsolcost<fitness(i)
            X(i,:) = newsolposition;
            fitness(i) = newsolcost;
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