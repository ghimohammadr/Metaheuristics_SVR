function [A,D,R,Params] = PhaseRecurr(x,varargin)
% -------------- [A,D,R,Params] = PhaseRecurr(x,varargin) ----------------
%
%   Calculates the phase-space representation and recurrence matrix 
%   of a time-series vector x using Taken's method of delays:
%
%       u(t) = x(t) + x(t + 1)tau + ... x(t + e-1)tau
%
%   where t = time, e = embedding dimension, tau = time delay.
%   Thus, the embedding dimension determines the demensions 
%   of vectors u(1) ... u(n-e+1), tau determines the dependence on previous
%   portions of the time series.
%
%   Typically, embedding dimensions are unknown, but can be estimated via
%   the method of false-nearest neighbors between vector u(t) and its
%   euclidean-defined nearest neighbor u_m(t) as:
%
%       |x(t + e) - x_m(t + e)| / D
%
%   where D = the distance between vectors u(t) and u_m(t). This
%   relationship is thresholded (default = 0.75) and the result is 0
%   or 1 indicating that the nearest neighbor distance D is true or false
%   due to the low-dimensional projection of time series x from its high-dimensional
%   dynamical manifold. 
%
%   Additionally, the time delay can also be estimated by the first local
%   minimum of the mutual information function between u(t) and u(t+tau)
%
%                   >>> INPUTS >>>
% Required:
%   x : time-series vector
% Optional:
%   tau     : the time delay to use for the phase-state reconstruction. Small
%                   tau are subject to noise, while large tau may smooth out dynamics.
%                   If not provided or empty, will search for tau associated with the
%                   first local minimum of the mutual information between
%                   u(t) and u(t+tau) for tau = 1:min(100,length(x)/2);
%   emb   : the embedding dimension. If not provided or empty,
%                   will estimate the best embedding dimension as described above
%   thresh: threshold for false-nearest neighbors (default = 0.75)
%   bin     : binsize for entropy calculations. If not provided or empty, will estimate
%                   best binsize by determining when rate of change in entropy is
%                   less than or equal to 5% max rate of change
%   epsilon: threshold value for recurrence matrix (between 0 and 1). If
%                   not provided or empty, a search algorithm will choose the epsilon
%                   that creates a recurrence matrix with 8-10% density
%                   (i.e. mean(sum(R) ./ size(R,1))), where each i,j in R is 0 or 1.
%
%                   <<< OUTPUTS <<<
%   A: the phase-space representation
%   D: the euclidean distance matrix from the phase space A
%   R: the recurrence matrix via epsilon threshold
%   Params: structure of parameters from estimations:
%       nbin - estimated (or given) bin size for entropy
%       tau  - estimated (or given) tau time delay
%       emb  - the estimated (or given) embedding dimension
%       err  - vector of percentage of false nearest neighbors starting with
%                  embedding dimension 3 to embedding dimension "emb"
%       eps  - estimated (or given) threshold for recurrence matrix
%       bin  - estimated (or given) bin size for entropy estimation
%       E    - entropy of x
%       P    - probability distribution of x
%
% By JMS, 4/1/2016
%------------------------------------------------------------------------------

% check inputs
%==================
if nargin > 1 && ~isempty(varargin{1})
    tau = varargin{1};
end
if nargin > 2 && ~isempty(varargin{2})
    emb = varargin{2}; 
end
if nargin > 3 && ~isempty(varargin{3})
    thresh = varargin{3};
else thresh = 0.75; end
if nargin > 4 && ~isempty(varargin{4})
    nbin = varargin{4};
else nbin = 0; end
if nargin > 5 && ~isempty(varargin{5})
    epsilon = varargin{5};
else epsilon = 0; end

if isrow(x); x = x'; end
%==================


% determine best time lag
% "tau" if not provided
%==================
if ~exist('tau','var')
    [tau,nbin] = estimate_tau(x,nbin);
end
%==================


% determine best embedding dimenson 
%  "e" if not provided
%==================
if ~exist('emb','var')
    [emb,err] = estimate_embedding(x,thresh,tau);
    err(err==0)=[];
else
    err = nan;
end
%==================


% construct the phase-space
%==================
A = phase_space(x,emb,tau);
%==================


% construct the distance matrix
%==================
if nargout >= 2
    D = find_dist(A);
end
%==================


% construct the recurrence matrix 
%==================
if nargout >= 3
    [R,epsilon] = get_recmat(D,epsilon); % R is single, since only 0 or 1
end
%==================


% store variables into "Params" structure
%==================
if nargout == 4

    % get entropy over entire "x"
    [E,P,nbin] = get_entropy(x,nbin);

    Params.tau = tau;
    Params.emb = emb;
    Params.err = err;
    Params.epsilon = epsilon;
    Params.bin = nbin;
    Params.E = E;
    Params.P = P;
end
%==================

end

%% Functions

% Entropy
%==========================
function [E,P,binsize] = get_entropy(d,bin)
    % calculates PDF and entropy from each column in matrix (vector) d
    % if 'bin' is 0 or empty, will loop through many bin-sizes for the histogram and
    % select the bin size that no longer causes significant changes in
    % entropy d. 
    
    ncol = size(d,2);
    nrow = size(d,1);
    
    % estimate bin size from entropy-vector derivative if bin == 0 or []
    if isempty(bin) || bin == 0
        fprintf('\nEstimating best entropy...\n')
        maxbin = floor(min(100)); % maximum bin size for loop
        if mod(maxbin,2) == 1
            maxbin = maxbin + 1;
        end
        Emat = zeros(maxbin/2,ncol); % since we are incrementing by 2 in the loop
        count = zeros(maxbin/2,1); % for getting bin count
        for b = 1:maxbin/2 % increment by 2...assuming 1-extra bin won't add much
            if mod(b,8)==0
                fprintf('%s ',' . ');
            end
            clear P
            [P,~] = hist(d,2*b);
            P = bsxfun(@rdivide,P,sum(P)); % sum(c) = 1
            Emat(b,:) = -sum(P .* log(P+eps)); % entropy...use nansum to avoid summing log(0)
            count(b) = 2*b; % store bin size
        end
        clear P;
        
        % get normalized difference of E w/ respect to max(diff(E))
        % and find first point where this is <= 5% (i.e. when rate of
        % change slows down to 5% of its max)
        z = bsxfun(@rdivide,diff(Emat),max(diff(Emat)));
        [ind,~] = find(z<=0.05,ncol,'first'); % gets first index for each column
        
        % now get final PDFs and entropy measurements at
        % the mean of the optimized bin indices
        binsize = round(mean(ind)) * 2;
        [P,~] = hist(d,binsize);
        P = bsxfun(@rdivide,P,sum(P));
        E = -sum(P .* log(P+eps));
        
    else % if bin is provided by the user
        [P,bins] = hist(d,bin);
        P = bsxfun(@rdivide,P,sum(P));
        E = -sum(P .* log(P+eps));
        binsize = bin;
    end
end  
%==========================


% Mututal Information
%==========================
function MI = mutual_info(E,P)
    % calculate joint entropy between probability P1 and Pm
    % by assuming first column in E,P represents probability to test
    % against remaining columns. Returns MI, with each number i 
    % representing mutual info between first and remaining columns
    
    ncol = numel(E)-1;
    
    % concatenate and get entropy and PDF for each column
    p1 = P(:,1); % probability of v
    pm = P(:,2:end); % probability of columns in m
    
    % make joint probability distribution for E_1 and E_m columns
    % and calculate joint entropy. Store into JE variable
    JE = zeros(1,ncol);
    for i = 1:ncol
        pj = p1 * pm(:,1)'; % joint probability 
        JE(i) = -sum(sum(pj .* log(pj+eps)));
    end
    
    % now calculate mutual information as:
    %   E_v + E_m - JE
    MI = E(1) + E(2:end) - JE;
end
%==========================
    

% Time-delay estimation
%==========================
function [tau,bin,mi] = estimate_tau(x,nbin)
    % estimates "tau" time-delay parameter via the first local
    % minimum of the mutual information between x and 
    % time lagged versions of x
    n = size(x,1);
    tmax = floor(min(100,2*n/3)); % maximum tau allowed
    T = zeros(n-tmax,tmax-1); % matrix will store data segments from x
     
    % construct T-matrix consisting of translated x vectors
    for t = 2:tmax+1
        T(:,t-1) = x(t:t+n-tmax-1);
    end
    x = x(1:1+n-tmax-1);
    
    % calculate entropy of each column in the system
    % if "nbin" = 0, "get_entropy" will estimate best bin size
    [E,P,bin] = get_entropy([x,T],nbin);
    
    % now get mutual information, assuming E(1) and P(:,1)
    % represent time-delay = 0;
    fprintf('\nEstimating tau...');
    M = mutual_info(E,P);
    coef = (1/7) * ones(1,7); % 7-point moving average coefficients
    Ms = filter(coef,1,M); % helps avoid peak detection of noise
    
    % get first local minimum index in mutual information
    [~,ix] = findpeaks(-Ms,'minpeakdistance',10); % arbitrarily set to 10...helps ensure true local minimum
    if isempty(ix)
        ix = 1;
    end
    
    % now extract tau, and min MI value 
    tau = ix(1); % first local minimum
    mi = M(tau);
   
   fprintf('\ntau is: %i\n',tau);
end
%==========================


% Embedding dimension estimation  
%==========================
function [est,err] = estimate_embedding(x,thresh,tau)
    % use a search algorithm to estimate the minimum embedding
    % dimension needed to represent high-dimensional phase-space of 
    % input vector x...we stop the loop when % false nearest-neighbors
    % falls below 1.

    minperc = 0.01; % 1% minimum false nearest neighbors 
    n = size(x,1); % num of data points
    emax = floor(min(60,(n/tau)-1)); % maximum embedding dimension to use
    err = zeros(emax,1); % make vector of zeros for storing percent error results

    %---- begin loop -----
    fprintf('\nEstimating embedding dimension...\n')
    i = 3; % embedding dimensions must be at least 2e +1, thus min = 3
    while i <= emax
        if mod(i,3) == 0
            fprintf('%s',' . ');
        end
        clear A dist h v ind dist2 ratio
        
        % -----extract phase space ------
        A = phase_space(x,i,tau);

        %----- find nearest-neighbors -------
        dist = zeros(size(A,1),2); % vector for storing results
        h = find_dist(A); % make distance matrix of all vectors in A
        h = h + diag(10000*ones(1,size(h,1)),0); % make i,i index for h arbitrarily large

        % get minimum distances and indices and store into "dist"
        [v,ind] = min(h);
        dist(:,1) = v'; 
        dist(:,2) = ind'; 

        % now check ratio of i+1 dim distance vs. i dim distance 
        if tau >1
            dist2 = abs(x(i+1:n-(i-1)*tau + i) - x(dist(:,2)+i)); % | x(t+e) - x_m(t+e) |
        else
            dist2 = abs(x(1+i:n) - x(dist(:,2)+i));
        end
        ratio = dist2 ./ dist(:,1); 

        % compare ratio to threshold...get percentage false nearest neighbors
        nfalse = ratio > thresh;
        perc = sum(nfalse) / numel(nfalse);
        err(i,1) = perc; 

        % check percentage, and break loop if critera has been met
        if perc <= minperc
            est = i;
            fprintf('\nembedding dimension is: %i\n',est);
            break
        end
        i = i+1;
    end % while loop
    if ~exist('est','var')
        est = i;
        fprintf('\ncould not converge embedding dimension...try using a smaller tau\n');
        fprintf('suboptimal dimension is: %i\n',est);
    end
end
%==========================


% Phase Space
%==========================
function A = phase_space(x,e,tau)
    % performs Taken's phase-space construction
    % with embedding dimension "e" and time-delay "tau"
    
    n = size(x,1);
    N = n - tau*(e-1);
    A = zeros(N,e); % maximum # of segments (k = 1,2 ... n - (e-1)*tau  
    
    % loop through series "x" and pull out e-dimensional vectors
    for t = 1:e
        A(:,t) = x([1:N] + tau*(t-1)); % indexes by tau
    end
end
%==========================


% Distance Matrix of Phase Space
%==========================
function D = find_dist(A)
    % takes in phase-space matrix "A" and calculates 
    % euclidean distances for all columns via vectorized notation 
  
    N = size(A,1); % size(x,1) - ((e-1)*tau)
    D = zeros(N,N); 
    
    % loop through A and get distances with other vectors in A
    for i = 1:N
        D(:,i) = sqrt(sum( bsxfun(@minus,A,A(i,:)).^2, 2));
    end
    
end
%==========================


% Recurrence matrix
%==========================
function [R,epsilon] = get_recmat(D,epsilon)
    % normalizes the distance matrix then thresholds by 
    % epsilon to get 1's or 0's corresponding to distances
    % smaller or larger than epsilon. Search for best epsilon to create
    % 8-10% density recurrence matrix
    
    D = D / max(max(D)); % normalize to [0,1] values
    N = size(D,1); 
    maxdensity = 0.1;
    mindensity = 0.08;
    
    % search for best epsilon if not provided
    if epsilon == 0;
        fprintf('\nEstimating recurrence epsilon...\n')
        SD = mean(std(D));
        Avg = mean(mean(D)); 
        guess = Avg*SD; % starting guess is 1 SD away from mean
        while epsilon == 0 % begin search
            count = 0;
            if mod(count,5) == 0
                fprintf(' . ');
            end
            if exist('density','var')
                density2 = density; % to compare pre-density
                clear R denstiy
            else
                density2 = 0;
            end

            R = D <= guess; % recurrence matrix
            density = mean(sum(R) / N); % density of recurrence matrix
            
            if density > maxdensity
                guess = guess - .01; 
            elseif density < mindensity && density2 < mindensity 
                guess = guess + .01;
            elseif density < mindensity && density2 > maxdensity
                % here the best guess is between density2 and density...
                % take the mean value between epsilon i and i-1
                epsilon = mean([guess,guess+.01]);
                fprintf('\nepsilon is: %s\n',epsilon);
            else
                epsilon = guess;
                fprintf('\nepsilon is: %s\n',epsilon);
                break
            end
            count = count+1;
        end
    else % if epsilon provided
        R = single(D <= epsilon);
    end
end