%
clear, clc
tic

% PARAMETERS
alpha= 0.33;
beta = .99; %discount factor 
sigma = 2; % coefficient of risk aversion
a_l = 0;
rho = 0.5;
sigma_eps=0.2;
n= 5; % starting point for rouwenhorst m= 5 grid points


% Z SPACE
% z = rho* ln(z)+e'

%rouwenhorst.m
%%[zgrid, P] = rouwenhorst(rho, sigma_eps, n)

zgrid=rouwenhorst(0.5,0.2,5)

% Find invariant distn, discretize grid with m =5 points


% ASSET VECTOR
a_min = a_l; %lower bound of grid points
a_max = 5; %upper bound of grid points - we guess
num_a = 500;
a = linspace(a_min, a_max, num_a); % asset (row) vector


% K VECTOR
num_k = 500;

% FIRMS PROBLEM
k_guess =  zeros(2, num_k);

% Solving FOC from Firm's problem, note= N=1
r = alpha*(k_guess)^(alpha-1)+1-delta
w = (1-alpha)*(k_guess^alpha)

% ITERATE OVER ASSET PRICES
aggsav = 1 ;
while abs(aggsav) >= 0.01 ;

% CURRENT RETURN (UTILITY) FUNCTION
% c = zwl+ra-a', l=1

cons = bsxfun(@minus, r *a, a');
cons = bsxfun(@plus, cons, permute(zgrid*w, [1 3 2])); 
ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
ret(cons<0) = -Inf;

% INITIAL VALUE FUNCTION GUESS
% v(z',a')
v_guess = zeros(2, num_a);

% VALUE FUNCTION ITERATION
v_tol = 1;
while v_tol >.0001;
   % CONSTRUCT TOTAL RETURN FUNCTION
   v_mat = ret + beta * ...
       repmat(permute(PI * v_guess, [3 2 1]), [num_a 1 1]);
   
   % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
   [vfn, pol_indx] = max(v_mat, [], 2);
   vfn = permute(vfn, [3 1 2]);
   
   v_tol = abs(max(v_guess(:) - vfn(:)));
   
   v_guess = vfn; %update value functions
   
end;

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
aggsav = sum( pol_fn(:) .* Mu(:) ); % Aggregate future assets

if aggsav > 0 ;
    q_min = q_guess ;
end ;
if aggsav < 0;
    q_max = q_guess ;
end ;




% Rouwenhorst function
% rouwenhorst.m
% rho is the 1st order autocorrelation
% sigma_eps is the standard deviation of the error term
% n is the number of points in the discrete approximation
%
function [zgrid, P] = rouwenhorst(rho,sigma_eps,n)

mu_eps = 0;

q = (rho+1)/2;
nu = ((n-1)/(1-rho^2))^(1/2) * sigma_eps;

P = [q 1-q;1-q q];


for i=2:n-1
   P = q*[P zeros(i,1);zeros(1,i+1)] + (1-q)*[zeros(i,1) P;zeros(1,i+1)] + ...
       (1-q)*[zeros(1,i+1); P zeros(i,1)] + q*[zeros(1,i+1); zeros(i,1) P];
   P(2:i,:) = P(2:i,:)/2;
end

zgrid = linspace(mu_eps/(1-rho)-nu,mu_eps/(1-rho)+nu,n);

end
