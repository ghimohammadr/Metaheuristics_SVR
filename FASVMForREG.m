%
% Firefly Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = FASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
alpha = 0.6;
betamin = 1;
lambda = 1;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];


%preallocation of firefly position and fitness
X = zeros(pop,part_dim); %pre_allocation of X firefly position
Xtemp = zeros(pop,part_dim);
fitness = zeros(pop,1); %pre_allocation of global fitness function (MSE) value

%%%%% Initialize particale position and evaluate initial fitness

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
Pbest_fit = fitness; %firefly personal best fitness
Gbest_position = X(gbestfitindex,:); %firefly globalbest position
Pbest_position = X; %firefly personal best position

for t = 1:max_iter %its ietration number
   Iteration = t
   
   delta=1-(10^(-4)/0.9)^(1/max_iter);
   alpha=(1-delta)*alpha;
   % Scaling 
    scale=abs(Ub-Lb);
    % Updating fireflies
    for i=1:pop,
    % The attractiveness parameter 
        for j=i:pop,
            r=sqrt(sum((X(i,:)-X(j,:)).^2));
            % Update moves
            if fitness(i)<Pbest_fit(j), % Brighter and more attractive
               
                beta = betamin*exp(-lambda*r.^2);
                tmpf=alpha.*(rand(1,part_dim)-0.5).*scale;
                Xtemp(i,:)=X(i,:)+(Pbest_position(j,:)-X(i,:)).*beta+tmpf;
                % being within limits
                Xtemp(i,:)= max(Xtemp(i,:), Lb);
                Xtemp(i,:)= min(Xtemp(i,:), Ub);
                %evaluation of fitness function 
                C = Xtemp(i,1);
                gam = Xtemp(i,2);
                epsil = Xtemp(i,3);
                
                svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
                model = svmtrain(Ytrain,Xtrain,svmoptions);
    
                [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
                
                MSEtemp = accuracy(2);
                if MSEtemp <fitness(i,1)
                    fitness(i,1) = accuracy(2);
                    X(i, :) = Xtemp(i,:);
                end
            end 
        end 
    end
    %update fireflies fitness
    [current_gbest_fit,current_index] = min(fitness);
    current_pbest_fit = fitness;
    %update global best fitness and position 
    if current_gbest_fit <= Gbest_fit
       Gbest_fit = current_gbest_fit;
       Gbest_position = X(current_index,:);
    end
   
    %update pesronal best fitness and position 
    Pbest_index = find(current_pbest_fit<=Pbest_fit);
    Pbest_position(Pbest_index,:) = X(Pbest_index,:);
    Pbest_fit(Pbest_index) = current_pbest_fit(Pbest_index);
end

c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save

end