%
% Cultural Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = CASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
VarSize=[1 part_dim];   % Decision Variables Matrix Size
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];
pAccept=0.35;                   % Acceptance Ratio
nAccept=round(pAccept*pop);    % Number of Accepted Individuals
alpha=0.3;
beta=0.5;
% Initialize Culture
CultureSituationalCost=inf;
CultureNormativeMin=inf(VarSize);
CultureNormativeMax=-inf(VarSize);
CultureNormativeL=inf(VarSize);
CultureNormativeU=inf(VarSize);

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
    
    fitness(i) = accuracy(2);
   
end

[~, SortOrder]=sort(fitness);
fitness = sort(fitness);
X = X(SortOrder);
Pbest_position = X; % personal best position


% Adjust Culture using Selected Population
sX=X(1:nAccept);

n=numel(sX);
nVar=numel(sX(1,:));
    
for i=1:n
    if fitness(i)<CultureSituationalCost
        CultureSituational=sX(i,:);
    end
        
    for j=1:nVar
        if sX(i,j)<CultureNormativeMin(j) ...
                || fitness(i)<CultureNormativeL(j)
            CultureNormativeMin(j)=sX(i,j);
            CultureNormativeL(j)=fitness(i);
        end
        if sX(i,j)>CultureNormativeMax(j) ...
                || fitness(i)<CultureNormativeU(j)
            CultureNormativeMax(j)=spop(i,j);
            CultureNormativeU(j)=fitness(i);
        end
    end
end
CultureNormativeSize=CultureNormativeMax-CultureNormativeMin;
    
% Update Best Solution Ever Found
BestSol=CultureSituational;

% Array to Hold Best Costs
BestCost=zeros(max_iter,1);

for t = 1:max_iter %its ietration number
     Iteration = t

    % Influnce of Culture
    for i=1:pop 
        
     % 3rd Method (using Normative and Situational components)
        for j=1:part_dim
            sigma=alpha*CultureNormativeSize(j);
            dx=sigma*randn;
            if X(i,j)<CultureSituationalPosition(j)
                dx=abs(dx);
            elseif X(i,j)>CultureSituationalPosition(j)
                dx=-abs(dx);
            end
            X(i,j)=X(i,j)+dx;
        end  
        
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
     
    [~, SortOrder]=sort(fitness);
    fitness = sort(fitness);
    X = X(SortOrder);
    
    % Adjust Culture using Selected Population
    sX=X(1:nAccept);

    n=numel(sX);
    nVar=numel(sX(1,:));
    
    for i=1:n
        if fitness(i)<CultureSituationalCost
            CultureSituational=sX(i,:);
        end
        
        for j=1:nVar
            if sX(i,j)<CultureNormativeMin(j) ...
                    || fitness(i)<CultureNormativeL(j)
                CultureNormativeMin(j)=sX(i,j);
                CultureNormativeL(j)=fitness(i);
            end
            if sX(i,j)>CultureNormativeMax(j) ...
                    || fitness(i)<CultureNormativeU(j)
                CultureNormativeMax(j)=spop(i,j);
                CultureNormativeU(j)=fitness(i);
            end
        end
    end
    CultureNormativeSize=CultureNormativeMax-CultureNormativeMin;
    
................. 
    
end

c = BestSol(1);
g = BestSol(2);
e = BestSol(3);
bestFitness = BestCost;
cost = model;
save

end
     