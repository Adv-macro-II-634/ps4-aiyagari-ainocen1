%ainocen1_tauchen.m

%
clear, clc
tic

% PARAMETERS
alpha= 0.33;
beta = .99; %discount factor 
sigma = 2; % coefficient of risk aversion
delta=0.025;


% Z SPACE
% z = rho* ln(z)+e'

%rouwenhorst.m
%%[zgrid, P] = rouwenhorst(rho, sigma_eps, n)

rho = 0.5;
sigma_eps=0.2;
n= 5; % starting point for rouwenhorst m= 5 grid points

% Find invariant distn, discretize grid with m =5 points
[zgrid,PI]=rouwenhorst(rho, sigma_eps, n);
z=exp(zgrid');

% PI is invariant if for stochastic matrix M, PI*M=PI
% (I-M)*PI = 0
tol=1;
while (tol >0.000001)
    PI_inv=PI*PI;
    tol=max(abs((PI_inv - PI)));
    PI=PI_inv;
end

N=z'*PI_inv(1,:)'; %I get N=1

% ASSET VECTOR
a_min = 0; %lower bound of grid points
a_max = 100; %upper bound of grid points - we guess
num_a = 500;
a = linspace(a_min, a_max, num_a); % asset (row) vector


% FIRMS PROBLEM
% K GUESS
k_min = 0;
k_max = 100;
k_guess = (k_min + k_max) / 2;


% ITERATE OVER ASSET PRICES
k_tol = 1 ;
limit=100; %setting max limit to break while loop
s=0;
tic
while abs(k_tol) >= 0.1 
   
    % Solving FOC from Firm's problem, note already imposed N=1
    k_guess = (k_min + k_max) / 2;
    r = alpha*(k_guess).^(alpha-1)+1-delta;
    w = (1-alpha)*(k_guess.^alpha);

    % CURRENT RETURN (UTILITY) FUNCTION
    % c = zwl+ra-a', l=1

    cons = bsxfun(@minus, r *a', a);
    cons = bsxfun(@plus, cons, permute(z'*w, [1 3 2])); 
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret(cons<0) = -Inf;

    % INITIAL VALUE FUNCTION GUESS
    % v(z',a')
    v_guess = zeros(n, num_a);

    % VALUE FUNCTION ITERATION
    v_tol = 1;
    
    while v_tol >.0001
       % CONSTRUCT TOTAL RETURN FUNCTION
       v_mat = ret + beta * ...
           repmat(permute(PI * v_guess, [3 2 1]), [num_a 1 1]);

       % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
       [vfn, pol_indx] = max(v_mat, [], 2);
       vfn = permute(vfn, [3 1 2]);

       v_tol = abs(max(v_guess(:) - vfn(:)));

       v_guess = vfn; %update value functions
       
       %Q = makeQmatrix(pol_indx, PI); for policy function iteration
       
    end



% KEEP DECSISION RULE
pol_indx = permute(pol_indx, [3 1 2]);
pol_fn = a(pol_indx);


% SET UP INITITAL DISTRIBUTION
Mu = zeros(n,num_a);
Mu(1, 4) = 1; %suppose full mass at one point

% ITERATE OVER DISTRIBUTIONS
mu_tol = 1;
while mu_tol > 1e-08
    % find Mu(z,a)
    [z_ind, a_ind, mass] = find(Mu > 0); % find non-zero indices

    MuNew = zeros(size(Mu));
    for ii = 1:length(z_ind)
        apr_ind = pol_indx(z_ind(ii), a_ind(ii)); 
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ...
            (PI(z_ind(ii), :) * Mu(z_ind(ii), a_ind(ii)) )';
    end

    mu_tol = max(abs(MuNew(:) - Mu(:)));

    Mu = MuNew ;
end


% CHECK AGGREGATE DEMAND
aggk = sum( pol_fn(:) .* Mu(:) ); %market clearing

if aggk > 0 
    k_min = k_guess ;
end 
if aggk < 0
    k_max = k_guess ;
end 

    temp=0;
    if temp>limit
        break
    end
    s=s+temp;
end
toc
% Since the model doesn't stop running, I put a break on the model (ctrl+c)
% (around 40 minutes)
% Then I run the commands below

%Plot policy function
plot(a,pol_fn);

% GINI COEFFICIENT
agg_wealth = aggk; % wealth is asset holdings plus incomes
wealth_dist = [[Mu(1,:), Mu(2,:),Mu(3,:),Mu(4,:),Mu(5,:)]; [r*a+z(1)*w,r*a+z(2)*w,r*a+z(3)*w,r*a+z(4)*w,r*a+z(5)*w]]';
[~, ordr] = sort(wealth_dist(:,2), 1);
wealth_dist = wealth_dist(ordr,:);

pct_dist = cumsum( (wealth_dist(:,2) ./ agg_wealth) .* wealth_dist(:,1) );
gini = 1 - sum( ([0; pct_dist(1:end-1)] + pct_dist) .* wealth_dist(:,1) );

display (['Gini coefficient of ', num2str(gini)]);


% r^(cm) = 1/beta = 1.0101
% resulting r = .9904, slightly less than r^(cm)

% Gini coefficient of -1.6572

