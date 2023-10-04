%
% Salp Swarm Optimization - SVR
%
function [cost,c,g,e,bestFitness] = SSASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
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
     
    c1 = 2*exp(-(4*(t+1)/max_iter)^2); 

    for i=1:pop
        
        X= X';
        
        %Update positions
        if i<=pop/2
            for j=1:1:part_dim
                c2=rand();
                c3=rand();
          
                if c3<0.5 
                    X(j,i)=Gbest_position(j)+c1*((Ub(j)-Lb(j))*c2+Lb(j));
                else
                    X(j,i)=Gbest_position(j)-c1*((Ub(j)-Lb(j))*c2+Lb(j));
                end
            
            end
            
        elseif i>pop/2 && i<pop+1
            point1=X(:,i-1);
            point2=X(:,i);
            
            X(:,i)=(point2+point1)/2; 
        end
        
        X= X';
        
        X(i,:)= max( X(i,:), Lb);
        X(i,:)= min( X(i,:),Ub);
    end
    
    for i=1:pop
                
        C = X(i,1);
        gam = X(i,2);
        epsil = X(i,3);
    
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);
    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
        fitness(i,1) = accuracy(2);        
    end

    [current_gbest_fit,current_index] = min(fitness);
    %update global best fitness and position 
    if current_gbest_fit <= Gbest_fit
       Gbest_fit = current_gbest_fit;
       Gbest_position = X(current_index,:);
    end

c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save

end



