%
% Particle Swarm Optimization - SVR
%
function [cost,c,g,e,bestFitness] = PSOSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

cvalue = [2.01,2.01];
pop = population;
c1 = cvalue(1);
c2 = cvalue(2);
max_iter = max_iteration;
part_dim = 3; %default value of dimension of particles
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];

%bound on velocity
Vcmax = (Crange(2)-Crange(1))/10; %velocity maximum bound on C
Vcmin = -Vcmax; %velocity minimum bound on C
Vgammax = (gamma(2)-gamma(1))/10; %velocity maximum bound on gamma
Vgammin = -Vgammax; %velocity minimum bound on gamma
Vepsmax = (epsilon(2)-epsilon(1))/10; %velocity maximum bound on epsilon
Vepsmin = -Vepsmax; %velocity minimum bound on epsilon
Vup = [Vcmax, Vgammax, Vepsmax];
Vdown = [Vcmin, Vgammin, Vepsmin];

%preallocation of particle position, velocity, pbest and fitness
X = zeros(pop,part_dim); %pre_allocation of X particle position
V = zeros(pop,part_dim); %pre_allocation of V velocity
Pbest_position = zeros(pop,part_dim); %pre_allocation of personal best position
fitness = zeros(pop,1); %pre_allocation of global fitness function (MSE) value

%%%%% Initialize particale position, velocity and evaluate initial fitness

for i=1:pop
    X(i,1) = Crange(1)+(Crange(2)-Crange(1)) * rand(1);
    X(i,2) = gamma(1)+(gamma(2)-gamma(1)) * rand(1);
    X(i,3) = epsilon(1)+(epsilon(2)-Crange(1)) * rand(1);
    V(i,1) = Vcmax+(Vcmax-Vcmin) * rand();
    V(i,2) = Vgammin+(Vgammax-Vgammin) * rand(1);
    V(i,3) = Vepsmin+(Vepsmax-Vepsmin) * rand(1);
    
    C = X(i,1);
    gam = X(i,2);
    epsil = X(i,3);
       
    svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
    model = svmtrain(Ytrain,Xtrain,svmoptions);    
    [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
    fitness(i,1) = accuracy(2);
    
end

[Gbest_fit, gbestfitindex] = min(fitness); %global fitness and its index value
Pbest_fit = fitness; %particle personal best fitness
Gbest_position = X(gbestfitindex,:); %particle globalbest position
Pbest_position = X; %particle personal best position

%update particle velocity and position

for t = 1:max_iter %its ietration number
   Iteration = t
   for i=1:pop
       
      w_up = 1.4;
      w_low = 0.8; 
      z = rand(1);
      mu = 4;
      z = mu * z *(1-z);
      w = (w_up - w_low) * ((max_iter - t)/max_iter) + w_low * z;
      
      V(i,:) = w * V(i,:) + c1* rand * (Pbest_position(i,:)-X(i,:)) + c2 * rand * (Gbest_position-X(i,:)); %update particle velocity
      
      %constrained to velocity upper and lower bound
      V(i,:) = min(V(i,:), Vup);
      V(i,:) = max(V(i,:), Vdown);
      
      %update particle current position
      X(i,:) = X(i,:) + V(i,:);
      %constrained to particle current position upper and lower boun      
      X(i,:) =min(X(i,:), Ub);
      X(i,:) = max(X(i,:), Lb);
      
      %evaluation of fitness function for update particles
      C = X(i,1);
      gam = X(i,2);
      epsil = X(i,3);

      svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
      model = svmtrain(Ytrain,Xtrain,svmoptions);    
      [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);

      fitness(i,1) = accuracy(2);

   end
   
   %update particles fitness
   [current_gbest_fit,current_index] = min(fitness);
   current_pbest_fit = fitness;
   %update global best fitness and position of particles
   if current_gbest_fit <= Gbest_fit
      Gbest_fit = current_gbest_fit;
      Gbest_position = X(current_index,:);
   end
   
   %update pesronal best fitness and position of particle
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