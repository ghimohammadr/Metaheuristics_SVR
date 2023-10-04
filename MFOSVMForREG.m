%
% Moth-Flame Optimization - SVR
%
function [cost,c,g,e,bestFitness] = MFOSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];

%preallocation of butterfly position and fitness
X = zeros(pop,part_dim); %pre_allocation of X butterfly position
fitness = zeros(pop,1); %pre_allocation of global fitness function (MSE) value

%%%%% Initialize position and evaluate initial fitness

for i=1:pop
    X(i,1) = Crange(1)+(Crange(2)-Crange(1)) * rand(1);
    X(i,2) = gamma(1)+(gamma(2)-gamma(1)) * rand(1);
    X(i,3) = epsilon(1)+(epsilon(2)-Crange(1)) * rand(1);
end

for ii = 1:max_iter %its ietration number
    Iteration = ii
     
    % Number of flames Eq. (3.14) in the paper
    Flame_no=round(pop-Iteration*((pop-1)/max_iter));
    
    for i=1:pop
        
        % Check if the simple limits/bounds are OK
        X(i,:)= max(X(i,:), Lb);
        X(i,:)= min(X(i,:),Ub);
        % Evaluate new solutions
        C = X(i,1);
        gam = X(i, 2);
        epsil = X(i, 3);
                
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);
    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
        fitness(i) = accuracy(2);
             
    end
    
    if Iteration==1
        % Sort the first population of moths
        [fitness_sorted I]=sort(fitness);
        sorted_population=X(I,:);
        
        % Update the flames
        best_flames=sorted_population;
        best_flame_fitness=fitness_sorted;
    else
        
        % Sort the moths
        double_population=[previous_population;best_flames];
        double_fitness=[previous_fitness; best_flame_fitness];
        
        [double_fitness_sorted I]=sort(double_fitness);
        double_sorted_population=double_population(I,:);
        
        fitness_sorted=double_fitness_sorted(1:pop);
        sorted_population=double_sorted_population(1:pop,:);
        
        % Update the flames
        best_flames=sorted_population;
        best_flame_fitness=fitness_sorted;
    end

    % Update the position best flame obtained so far
    Best_flame_score=fitness_sorted(1);
    Best_flame_pos=sorted_population(1,:);
      
    previous_population=X;
    previous_fitness=fitness;
    
    % a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a=-1+Iteration*((-1)/max_iter);   
    
    for i=1:pop
        
        for j=1:part_dim
            if i<=Flame_no % Update the position of the moth with respect to its corresponsing flame
                
                % D in Eq. (3.13)
                distance_to_flame=abs(sorted_population(i,j)-X(i,j));
                b=1;
                t=(a-1)*rand+1;
                
                % Eq. (3.12)
                X(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(i,j);
            end
            
            if i>Flame_no % Upaate the position of the moth with respct to one flame
                
                % Eq. (3.13)
                distance_to_flame=abs(sorted_population(i,j)-X(i,j));
                b=1;
                t=(a-1)*rand+1;
                
                % Eq. (3.12)
                X(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(Flame_no,j);
            end
        end
    end
    
end

c = Best_flame_pos(1);
g = Best_flame_pos(2);
e = Best_flame_pos(3);
bestFitness = Best_flame_score;
cost = model;
save

end
     
