%
% Golden Sine Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = GSASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];
a=-pi;                           
b=pi;
gold=double((sqrt(5)-1)/2);      % golden proportion coefficient, around 0.618
x1=a+(1-gold)*(b-a);          
x2=a+gold*(b-a);

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

for t = 1:max_iter %its ietration number
     Iteration = t
     
    % Update the position of solutions with respect to objective
    for i=1:pop % in i-th solution
        r=rand;
        r1=(2*pi)*r;
        r2=r*pi; 
        for j=1:part_dim % in j-th dimension
           
            X(i,j)= X(i,j)*abs(sin(r1)) - r2*sin(r1)*abs(x1*Gbest_position(j)-x2*X(i,j));
            
        end
    end
    
    for i=1:pop
        
        % Check if solutions go outside the search spaceand bring them back
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

        % If fitness improves (better solutions found), update then
        if (fitness(i)<Gbest_fit),
            Gbest_position=X(i,:);
            Gbest_fit=fitness(i);
            b=x2;
            x2=x1;
            x1=a+(1-gold)*(b-a);
        else
            a=x1;
            x1=x2;
            x2=a+gold*(b-a);
        end
                        
        if x1==x2
            a=-pi*rand; 
            b=pi*rand;
            x1=a+(1-gold)*(b-a); 
            x2=a+gold*(b-a);
        end
        
    end

c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save
    
end
     

