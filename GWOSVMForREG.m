%
% Grey Wolf Optimization Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = GWOSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,part_dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,part_dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,part_dim);
Delta_score=inf; %change this to -inf for maximization problems

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

Convergence_curve=zeros(1,max_iter);

for t = 1:max_iter %its ietration number
    Iteration = t
 
    for i=1:pop  
        
       % Return back the search agents that go beyond the boundaries of the search space
%        Flag4ub=X(i,:)>ub;
%        Flag4lb=X(i,:)<lb;
%        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               
        
        % Calculate objective function for each search agent
        C = X(i,1);
        gam = X(i, 2);
        epsil = X(i, 3);
                
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);
    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
        fitness(i) = accuracy(2);
        
        % Update Alpha, Beta, and Delta
        if fitness(i)<Alpha_score 
            Alpha_score=fitness(i); % Update alpha
            Alpha_pos=X(i,:);
        end
        
        if fitness(i)>Alpha_score && fitness(i)<Beta_score 
            Beta_score=fitness(i); % Update beta
            Beta_pos=X(i,:);
        end
        
        if fitness(i)>Alpha_score && fitness(i)>Beta_score && fitness(i)<Delta_score 
            Delta_score=fitness(i); % Update delta
            Delta_pos=X(i,:);
        end
    end

    a=2-t*((2)/max_iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:pop
        for j=1:part_dim     
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-X(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-X(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-X(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            X(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
        
        X(i,:) = max(X(i,:), Lb);
        X(i,:) = min(X(i,:), Ub);
    end
    
    Convergence_curve(t)=Alpha_score;


end

c = Alpha_pos(1);
g = Alpha_pos(2);
e = Alpha_pos(3);
bestFitness = Alpha_score;
cost = model;
save

end
         
         
         