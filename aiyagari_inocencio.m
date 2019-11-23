%
clear, clc
tic

% PARAMETERS
alpha= 0.33;
beta = .99; %discount factor 
sigma = 2; % coefficient of risk aversion
delta=0.025;
a_l = 0;



% Z SPACE
% z = rho* ln(z)+e'

%rouwenhorst.m
%%[zgrid, P] = rouwenhorst(rho, sigma_eps, n)

rho = 0.5;
sigma_eps=0.2;
n= 5; % starting point for rouwenhorst m= 5 grid points

[zgrid,PI]=rouwenhorst(rho, sigma_eps, n);

% Find invariant distn, discretize grid with m =5 points


% ASSET VECTOR
a_min = a_l; %lower bound of grid points
a_max = 10; %upper bound of grid points - we guess
num_a = 500;
a = linspace(a_min, a_max, num_a); % asset (row) vector




% FIRMS PROBLEM
% K GUESS
k_min = 0;
k_max = 1;
k_guess = (k_min + k_max) / 2;


% Solving FOC from Firm's problem, note= N=1
r = alpha*(k_guess).^(alpha-1)+1-delta;
w = (1-alpha)*(k_guess.^alpha);

% ITERATE OVER ASSET PRICES
aggsav = 1 ;
while abs(aggsav) >= 0.01 

% CURRENT RETURN (UTILITY) FUNCTION
% c = zwl+ra-a', l=1

cons = bsxfun(@minus, r *a, a');
cons = bsxfun(@plus, cons, permute(zgrid*w, [1 3 2])); 
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

end



% KEEP DECSISION RULE
pol_indx = permute(pol_indx, [3 1 2]);
pol_fn = a(pol_indx);


% SET UP INITITAL DISTRIBUTION
Mu = zeros(2,num_a);
Mu(1, 4) = 1; % initial guess: everyone employed, 0 assets
% Mu = ones(2, num_a); alternative initial guess: same mass in all states
% Mu = Mu_guess / sum(Mu_guess(:)); % normalize total mass to 1


% ITERATE OVER DISTRIBUTIONS
% way 1: loop over all non-zeros states
mu_tol = 1;
while mu_tol > 1e-08
    [emp_ind, a_ind] = find(Mu > 0); % find non-zero indices

    MuNew = zeros(size(Mu));
    for ii = 1:length(emp_ind)
        apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); 
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ...
            (PI(emp_ind(ii), :) * Mu(emp_ind(ii), a_ind(ii)) )';
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

end

