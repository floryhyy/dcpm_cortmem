function [y_predict, performance, final_net, mean_net] = cpm_singlefunc(x,y,varargin)
dbstop if error
% (from cpm_main)
% Performs Connectome-Based Predictive Modeling (CPM)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        x            Predictor variable (e.g., connectivity matrix)
%                    Allowed dimensions are 2D (n x nsubs) OR 3D (nx m x nsubs)
%        y            Outcome variable (e.g., behavioral scores)
%                    Allowed dimensions are 2D (i x nsubs)
%        'pthresh'    p-value threshold for feature selection
%        'kfolds'     Number of partitions for dividing the sample
%                    (e.g., 2 =split half, 10 = ten fold)
%        'id'        Vector of subject ids (nsubs x 1) to leave out
%        subject. Otherwise, assumes vector 1:n
%
%   OUTPUTS
%        y_predict    Predictions of outcome variable
%        performance  Correlation between predicted and actual values of y
%
%   Example:
%        [yhat,perf]=cpm_main(data,gF,'pthresh',0.05,'kfolds',2);
%
%   References:
%        If you use this script, please cite:
%        Shen, X., Finn, E. S., Scheinost, D., Rosenberg, M. D., Chun, M. M.,
%          Papademetris, X., & Constable, R. T. (2017). Using connectome-based
%          predictive modeling to predict individual behavior from brain connectivity.
%          Nature Protocols, 12(3), 506.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dbstop if error;
%% Parse input
p=inputParser;
defaultpthresh=0.01;
defaultkfolds=2; 
defaultid=[];

addRequired(p,'x',@isnumeric);
addRequired(p,'y',@isnumeric); % must be n x nsubs
addParameter(p,'pthresh',defaultpthresh,@isnumeric);
addParameter(p,'kfolds',defaultkfolds,@isnumeric);
addParameter(p,'id',defaultid,@isnumeric);
addParameter(p,'categorical',defaultid,@islogical);
addParameter(p,'flatten',defaultid,@islogical);


parse(p,x,y,varargin{:});

pthresh = p.Results.pthresh;
kfolds = p.Results.kfolds;
id = p.Results.id;
categorical=p.Results.categorical;
flatten=p.Results.flatten;


%%% evg mod: if no id list provided, just treat each data point as independent %%%
loo_list = 1:size(x,2);
if (isempty(id))
    id = loo_list;
end
%%% end mod

clearvars p

%% Check for errors - evg mod to store number of nodes
[x,y,no_nodes, symm_status]=cpm_check_errors(x,y,kfolds,id);

%% Train & test Connectome-Based Predictive Model
[y_predict, final_net, mean_net]=cpm_cv_subj(x,y,id,pthresh,kfolds,no_nodes,categorical,flatten, symm_status);

%% Assess performance
[performance.mean.r_rank(1),performance.mean.r_rank(2)]=corr(y_predict.mean(:),y(:), 'type', 'spearman');
[performance.mean.r_pearson(1),performance.mean.r_pearson(2)]=corr(y_predict.mean(:),y(:), 'type', 'pearson');

[performance.neg.r_rank(1),performance.neg.r_rank(2)]=corr(y_predict.neg(:),y(:), 'type', 'spearman');
[performance.neg.r_pearson(1),performance.neg.r_pearson(2)]=corr(y_predict.neg(:),y(:), 'type', 'pearson');

[performance.pos.r_rank(1),performance.pos.r_rank(2)]=corr(y_predict.pos(:),y(:), 'type', 'spearman');
[performance.pos.r_pearson(1),performance.pos.r_pearson(2)]=corr(y_predict.pos(:),y(:), 'type', 'pearson');


performance.mean.mse = sum((y_predict.mean-y).^2)/length(y);
performance.neg.mse = sum((y_predict.neg-y).^2)/length(y);
performance.pos.mse = sum((y_predict.pos-y).^2)/length(y);

performance.mean.q_s = 1-performance.mean.mse/var(y_predict.mean,1);
performance.neg.q_s = 1-performance.mean.mse/var(y_predict.neg,1);
performance.pos.q_s = 1-performance.mean.mse/var(y_predict.pos,1);

    % evg mod: regress out motion from behavior in each cv fold
%[performance(1),performance(2)]=corr(y_predict(:),(y(:)./mot(:)));
    fprintf('\nDone.\n')
end

%% >>>>> SUPPORTING FUNCS <<<<< %%

%% CHECK INPUTS
function [x, y, no_nodes, symm_status] = cpm_check_errors(x, y, folds, id)
% Checks that input data is in a format usable by CPM

% Check that x data are in the required format - 2D or 3D
% if (ndims(x)~=2) && (ndims(x)~=3)
%    error('Data should have two or three dimensions') 
% end

% Check that x data contain more than one element
if size(x,1)==1
    error('Single feature detected.')
end

%% evg mod - store number of nodes

if (ndims(x)==2)
    syms n;
    eqn= n^2-n==size(x,1)*2;
    S = solve(eqn);
    no_nodes=double(S(2));
elseif (ndims(x)==1)
    no_nodes = [size(x,1),1];
elseif (ndims(x)==3)
    no_nodes=size(x,1);
end

% Check that there the same number of subjects in x as in y
if size(y,1)==1 % If behavioral scores are row vector, reformat to be column vector
    y = y';
end

if size(x,ndims(x))~=size(y,1)
    error('There are NOT the same number of subjects in the data and behavior variable')
end

% Check to make sure there are at least ten subjects in the input data
if size(x,ndims(x))<10
    warning('The CPM code requires >10 subjects to function properly; sound results likely require >>10.')
end

% Check to make sure you have more subjects than folds
if size(x,ndims(x))<folds
    warning('You must have more subjects than folds in your cross validation. Please check the help documentation.')
end

% Check whether x is symmetric across first two dimensions
if ndims(x)==3
    if size(x,1)~=size(x,2)
        warning('Please make sure, if intended, that data is an NxN connectivity matrix')
    end
end

% Check for nodes with values of 0 (missing nodes within a subject)
row_sum = squeeze(sum(abs(x), 2));
zero_node = sum(row_sum==0);
zero_node_subjects = sum( zero_node>0);
if zero_node_subjects>0
    warning('Data: %d subjects have missing nodes. Please check your data.',zero_node_subjects)
end

% Check for Inf or NaN 
if length(find(isinf(x)))>0
    warning('You have Inf values in your matrices. Please check your data.')
end

if length(find(isnan(x)))>0
    warning('You have NaNs in your matrices. Please check your data.')
end

% evg mod: if leaving one subject out, make sure that subject list is
% included
loo_list = 1:size(x,3);
if ~isequal(id,loo_list)
    warning('Assuming multiple values per subject. Make sure correct subject list is provided')
else
    warning('Leaving single value out per loop. No subject list provided')
end

% If data are 3D, convert to 2D
% If matrix is symmetric, only upper triangle is taken
if ndims(x)==3
    if issymmetric(x(:,:,1))
        symm_status = 1;
        s=size(x,1);
        for i=1:size(x,3)
            data_tmp=x(:,:,i);
            m(:,i)=data_tmp(logical(triu(ones(s,s),1)));
        end
    else
        symm_status = 0;
        m=reshape(x,size(x,1)*size(x,2),size(x,3));
    end
    x=m;
else 
    symm_status=0;
end
end

%% CV
function [y_predict, final_net, mean_net]=cpm_cv_subj(x,y,id,pthresh,kfolds,no_nodes,categorical,flatten,symm_status)
% Runs cross validation for CPM
% x            Predictor variable
% y            Outcome variable
% id           List of subject IDs **evg addition**
% mot          Motion **evg addition**
% pthresh      p-value threshold for feature selection
% kfolds       Number of partitions for dividing the sample
% y_test       y data used for testing
% y_predict    Predictions of y data used for testing
% no_nodes     Number of nodes in connectome (for saving significant matrices) **evg addition** 
% symm_status  If matrix is symmetrical, converts back to 377 x 377 grid
%              **evg addition**
 
% Split data
ndata=size(x,2); %multiple observations per sub
y_predict.mean = zeros(ndata, 1);
y_predict.pos = zeros(ndata, 1);
y_predict.neg = zeros(ndata, 1);

%%% evg mod: to store networks %%%
edges_pos_all=ones(size(x,1),1);    edges_neg_all=ones(size(x,1),1);

% leaving out entire subject(s)
id_list = unique(id);
nsub = length(id_list);
randinds = id_list(randperm(length(id_list)));
ksample = floor(nsub/kfolds); %<CHECK
     
% Run CPM over all folds
fprintf('\n# Running over %1.0f folds.\nPerforming fold no. ',kfolds);
for leftout = 1:kfolds
    fprintf('%1.0f ',leftout);
    
    if kfolds == nsub % doing leave-one-subject-out
        testinds=randinds(leftout);
        traininds=setdiff(randinds,testinds);
    else
        si=1+((leftout-1)*ksample);
        fi=si+ksample-1;
        
        testinds=randinds(si:fi);
        traininds=setdiff(randinds,testinds);
    end
    
    % Assign x and y data to train and test groups 
    %%%% evg mod: based on subjects %%%%
    x_train = x(:,ismember(id, traininds));
    y_raw = y(ismember(id, traininds));
    x_test = x(:,ismember(id, testinds));
    
    
    
    %%%% evg mod: regress out motion from behavior %%%%
%     if size(mot,1)==1 % If behavioral scores are row vector, reformat to be column vector
%     mot = mot';
%     end

%   y_train = (y_raw./mot(ismember(id, traininds)));
      y_train = y_raw;     
     
    % Train Connectome-based Predictive Model
    [~, ~, pmask, mdl] = cpm_train(x_train, y_train, pthresh,categorical);

%%%% evg mod: store the edges that are predictive on every loop %%%%
    edges_pos = pmask == 1;
    edges_neg = pmask == -1;
    edges_pos_all = edges_pos_all.*edges_pos;
    edges_neg_all = edges_neg_all.*edges_neg;

    % Test Connectome-based Predictive Model
    [y_predict_output] = cpm_test(x_test,mdl,pmask,categorical);
    y_predict.mean(ismember(id, testinds))= y_predict_output.mean;
    y_predict.pos(ismember(id, testinds))= y_predict_output.pos;
    y_predict.neg(ismember(id, testinds))= y_predict_output.neg;


    
end % folds

%%%% evg mod: convert edges selected on each loop back to matrix %%%%

if flatten
    final_net.pos = edges_pos_all;
    final_net.neg = edges_neg_all;
else
    
pos_mat=zeros(no_nodes);

if symm_status == 1
pos_mat(triu(true(no_nodes),1))=edges_pos_all;
pos_mat=pos_mat+pos_mat.' - diag(diag(pos_mat));
end

neg_mat=zeros(no_nodes); 

if symm_status == 1
neg_mat(triu(true(no_nodes),1))=edges_neg_all;
neg_mat=neg_mat+neg_mat.'- diag(diag(neg_mat));
end

final_net.pos = pos_mat;
final_net.neg = neg_mat;
end


% and save mean value of network per participant
mean_net.pos = mean(x(edges_pos_all==1,:));
mean_net.neg = mean(x(edges_neg_all==1,:));

end

%% TRAIN
function [r,p,pmask,mdl]=cpm_train(x,y,pthresh,categorical)
% Train a Connectome-based Predictive Model
% x            Predictor variable
% y            Outcome variable
% pthresh      p-value threshold for feature selection
% r            Correlations between all x and y
% p            p-value of correlations between x and y
% pmask        Mask for significant features
% mdl          Coefficient fits for linear model relating summary features to y

% Select significant features

[r,p]=corr(x',y);
pmask_tmp=(+(r>0))-(+(r<0));
pmask_tmp=pmask_tmp.*(+(p<pthresh));

% >> if none survive, rank and pick top X most distinctive (up to "elbow")
if ~any(pmask_tmp==1)
    % convert to z score
    rtoztrans = @(x) 0.5*log((1+x)/(1-x));
    z = arrayfun(rtoztrans, r);
    zidx = find(r>0);
    [~,desc] = sort(r(r>0),'descend');
    zsort = z(zidx(desc));
    [~,maxidx] = max(abs(diff(zsort)));
    topX = zsort(1:maxidx);
    topXmask = zeros(size(pmask_tmp,1), 1);
    topXmask(find(ismember(z, topX))) = 1;

    z = arrayfun(rtoztrans, r);
    zidx = find(r<0);
    [~,desc] = sort(r(r<0),'descend');
    zsort = z(zidx(desc));
    [~,maxidx] = max(abs(diff(zsort)));
    topX = zsort(1:maxidx);
    topXmask = zeros(size(pmask_tmp,1), 1);
    topXmask(find(ismember(z, topX))) = -1;

    pmask = topXmask;
else 
    pmask= pmask_tmp;
end

% For each subject, summarize selected features 
for i=1:size(x,2)
    summary_feature.mean(i)=nanmean(x(pmask>0,i))-nanmean(x(pmask<0,i));
    summary_feature.pos(i)=nanmean(x(pmask>0,i));
    summary_feature.neg(i)=-nanmean(x(pmask<0,i));
end

% flory edit: if no edges are selected
if sum(isnan(summary_feature.mean))==length(summary_feature.mean)
    mdl.mean=[0;0];
else
% Fit y to summary features
    if categorical
       [B,dev,stats]=mnrfit(summary_feature.mean,y,'model','ordinal');
        mdl.mean=B;
    else
        mdl.mean=glmfit(summary_feature.mean,y,'binomial','link','logit');
    end
    
end

if sum(isnan(summary_feature.pos))==length(summary_feature.pos)
    mdl.pos=[0;0];
else
    if categorical
       [B,dev,stats]=mnrfit(summary_feature.pos,y,'model','ordinal');
        mdl.pos=B;
    else
        mdl.pos=glmfit(summary_feature.pos,y,'binomial','link','logit');
    end
end

if sum(isnan(summary_feature.neg))==length(summary_feature.neg)
    mdl.neg=[0;0];
else
    if categorical
       [B,dev,stats]=mnrfit(summary_feature.neg,y,'model','ordinal');
       mdl.neg =B;
    else
        mdl.neg=glmfit(summary_feature.neg,y,'binomial','link','logit');
    end
end

end

%% TEST
function [y_predict]=cpm_test(x,mdl,pmask,categorical)
% Test a Connectome-based Predictive Model using previously trained model
% x            Predictor variable
% mdl          Coefficient fits for linear model relating summary features to y
% pmask        Mask for significant features
% y_predict    Predicted y values

% For each subject, create summary feature and use model to predict y
for i=1:size(x,2)
    summary_feature.mean(i)=nanmean(x(pmask>0,i))-nanmean(x(pmask<0,i));
    
    summary_feature.pos(i)=nanmean(x(pmask>0,i));
    summary_feature.neg(i)=-nanmean(x(pmask<0,i));
    if categorical
        [ max_value, max_index ]=max(mnrval(mdl.mean,summary_feature.mean(i),'model','ordinal'));
        y_predict.mean(i)=max_index;
        [ max_value, max_index ]=max(mnrval(mdl.pos,summary_feature.pos(i),'model','ordinal'));
        y_predict.pos(i)=max_index;
        [max_value, max_index ]=max(mnrval(mdl.neg,summary_feature.neg(i),'model','ordinal'));
        y_predict.neg(i)=max_index;
    else
        y_predict.mean(i)=glmval(mdl.mean,summary_feature.mean(i),'logit')>0.5;
        y_predict.pos(i)=glmval(mdl.pos,summary_feature.pos(i),'logit')>0.5;
        y_predict.neg(i)=glmval(mdl.neg,summary_feature.neg(i),'logit')>0.5;
    end
end

end

