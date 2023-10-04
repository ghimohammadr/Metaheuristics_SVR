%
% Whale Optimization Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = WOASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

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
end

% initialize position vector and score for the leader
Leader_pos=zeros(1,part_dim);
Leader_score=inf; %change this to -inf for maximization problems

for t = 1:max_iter %its ietration number
     Iteration = t
     
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
        fitness = accuracy(2);
        
        % Update the leader
        if fitness<Leader_score % Change this to > for maximization problem
            Leader_score=fitness; % Update alpha
            Leader_pos=X(i,:);
        end  
    end
    
    a=2-t*((2)/max_iter); % a decreases linearly fron 2 to 0 in Eq. (2.3)
    
    % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a2=-1+t*((-1)/max_iter);
    
    % Update the Position of search agents 
    for i=1:pop
        
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]        
        A=2*a*r1-a;  % Eq. (2.3) in the paper
        C=2*r2;      % Eq. (2.4) in the paper               
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)        
        p = rand();        % p in Eq. (2.6)
        
        for j=1:part_dim       
            
            if p<0.5   
                
                if abs(A)>=1
                    
                    rand_leader_index = floor(pop*rand()+1);
                    X_rand = X(rand_leader_index, :);
                    D_X_rand=abs(C*X_rand(j)-X(i,j)); % Eq. (2.7)
                    X(i,j)=X_rand(j)-A*D_X_rand;      % Eq. (2.8)                   
                else
                    
                    D_Leader=abs(C*Leader_pos(j)-X(i,j)); % Eq. (2.1)
                    X(i,j)=Leader_pos(j)-A*D_Leader;      % Eq. (2.2)
                end  
                
            else
                distance2Leader=abs(Leader_pos(j)-X(i,j));
                % Eq. (2.5)
                X(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j);
            end            
        end
    end
end

c = Leader_pos(1);
g = Leader_pos(2);
e = Leader_pos(3);
bestFitness = Leader_score;
cost = model;
save 
  
end
     
     
     
     
