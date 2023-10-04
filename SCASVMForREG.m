%
% Sine Cosine Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = SCASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];


%preallocation of position and fitness
X = zeros(pop,part_dim); %pre_allocation of X position
Pbest_position = zeros(pop,part_dim); %pre_allocation of personal best position
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
       
    a = 2;
    r1=a-t*((a)/max_iter); % r1 decreases linearly from a to 0
    
    % Update the position of solutions with respect to destination
    for i=1:pop % in i-th solution
        for j=1:part_dim % in j-th dimension
            
                        % Update r2, r3, and r4 
            r2=(2*pi)*rand();
            r3=2*rand;
            r4=rand();

            if r4<0.5

                Pbest_position(i,j)= X(i,j)+(r1*sin(r2)*abs(r3*Gbest_position(j)-X(i,j)));
            else
                
                Pbest_position(i,j)= X(i,j)+(r1*cos(r2)*abs(r3*Gbest_position(j)-X(i,j)));
            end
        end
    end

    for i=1:pop
         
        % Check if solutions go outside the search space and bring them back
        Pbest_position(i,:)= max(Pbest_position(i,:), Lb);
        Pbest_position(i,:)= min(Pbest_position(i,:), Ub);
        
        % Calculate the fitness values
        C = Pbest_position(i,1);
        gam = Pbest_position(i,2);
        epsil = Pbest_position(i,3);
                
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);
    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
                
        MSEtemp = accuracy(2);        
        if MSEtemp <fitness(i,1)
             fitness(i,1) = accuracy(2);
             X(i, :) = Pbest_position(i,:);
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