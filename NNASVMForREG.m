%
% Neural network algorithm - SVR
%
function [cost,c,g,e,bestFitness] = NNASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];
beta=1;

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

% Creat random initial weights with constraint of Summation each column = 1
ww=ones(1,pop)*0.5;
w=diag(ww);
for i=1:pop
    t=rand(1,pop-1)*0.5;
    t=(t./sum(t))*0.5;
    w(w(:,i)==0,i)=t;
end

[Gbest_fit, gbestfitindex] = min(fitness); %global fitness and its index value
Gbest_position = X(gbestfitindex,:); % globalbest position
wtarget=w(:,gbestfitindex);           % Best obtained weight (weight target)

FMIN=zeros(max_iter,1);

for t = 1:max_iter %its ietration number
     Iteration = t
     
    %------------------ Creating new solutions ----------------------------
    x_new=w*X;
    X=x_new+X;
    %------------------- Updating the weights -----------------------------
    for i=1:pop
        w(:,i)=abs(w(:,i)+((wtarget-w(:,i))*2.*rand(pop,1)));
    end
    
    for i=1:pop
        w(:,i)=w(:,i)./sum(w(:,i));    % Summation of each column = 1
    end
    
    for i=1:pop
        
        if rand<beta
            
            %------------- Bias for input solutions -----------------------
            N_Rotate=ceil(beta*part_dim);
            
            xx=Lb+(Ub-Lb).*rand(1,part_dim);
            rotate_postion=randperm(part_dim);rotate_postion=rotate_postion(1:N_Rotate);
            
            for m=1:N_Rotate
                X(i,rotate_postion(m))=xx(m);
            end
            %---------- Bias for weights ----------------------------------
            N_wRotate=ceil(beta*pop);
            
            w_new=rand(N_wRotate,pop);
            rotate_position=randperm(pop);rotate_position=rotate_position(1:N_wRotate);
            
            for j=1:N_wRotate
                w(rotate_position(j),:)=w_new(j,:);
            end
            
            for iii=1:pop
                w(:,iii)=w(:,iii)./sum(w(:,iii));   % Summation of each column = 1
            end
        else
            %------------ Transfer Function Operator ----------------------
            X(i,:)=X(i,:)+(Gbest_position-X(i,:))*2.*rand(1,part_dim);
        end
        
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

    
    [current_gbest_fit,current_index] = min(fitness);
    %update global best fitness and position 
    if current_gbest_fit <= Gbest_fit
       Gbest_fit = current_gbest_fit;
       Gbest_position = X(current_index,:);
       wtarget=w(:,current_index);
    end
    
    % ---------------------- Bias Reduction -------------------------------
    beta=beta*0.99;
    if beta<0.01
        beta=0.05;
    end
    
end

c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save   
    
end



