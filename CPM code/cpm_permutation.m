%% Run permutation testing for CPM analysis - this file runs on the cluster
% This script performs permutation testing (1000 iterations) on CPM models for recognition 
% and arousal across different stimuli and pill conditions

% Initialize parallel pool
poolobj = parpool;
fprintf('Number of workers: %g\n', poolobj.NumWorkers);

% Define analysis parameters
pheno_ls = ["RecogAcc_coded" "Arous_All"];
stim_ls = ["alc" "tool"];
pill_ls = ["cortisol" "placebo"];
folder = "data/";
p = 0.01;
permutation = true;
n = 1000;  % number of permutations

for pheno = pheno_ls
   for stim = stim_ls
       for pill = pill_ls
           % Load data
           filename = append(folder, 'n26_', pheno, '_', pill, '_', stim, '_trialID_dynamic_connectome.mat');
           data = load(filename);
           x = [data.X];
           x = permute(x, [2 3 1]);
           y = [data.y]';
           id = [data.id];
           
           % Set output file path based on phenotype
           if pheno == "RecogAcc_coded"
               outputfile = append('result/permutation/5s_onward_BPfiltered/', pheno, '_byTrial_', pill, '_', stim, '_p0.01.mat');
               categorical = false;
           else
               outputfile = append('result/permutation/5s_onward_BPfiltered/linear_', pheno, '_byTrial_', pill, '_', stim, '_p0.01.mat');
               categorical = true;
           end
           
           % Run actual model
           [y_predict, performance, final_net, mean_net] = cpm_singlefunc(x, y, 'kfolds', 40, 'id', id, 'flatten', false, 'p', p);
           test.performance = performance;
           test.y_predict = y_predict;
           
           if permutation
               % Initialize arrays for permutation results
               null_mean_r_rank = zeros(n, 1);
               null_mean_r_pearson = zeros(n, 1);
               null_pos_r_rank = zeros(n, 1);
               null_pos_r_pearson = zeros(n, 1);
               null_neg_r_rank = zeros(n, 1);
               null_neg_r_pearson = zeros(n, 1);
               null_mean_y_predict = zeros(n, size(y, 1));
               null_pos_y_predict = zeros(n, size(y, 1));
               null_neg_y_predict = zeros(n, size(y, 1));
               
               hold on;
               parfor i = 1:n
                   % Shuffle labels
                   shuffled_y = y(randperm(size(y, 1)));
                   
                   % Run CPM on shuffled data
                   [null_y_predict, null_performance, ~, ~] = cpm_singlefunc(x, shuffled_y, 'kfolds', 40, 'id', id, 'flatten', false, 'p', p);
                   
                   % Store results
                   null_mean_r_rank(i) = null_performance.mean.r_rank(1);
                   null_mean_r_pearson(i) = null_performance.mean.r_pearson(1);
                   null_pos_r_rank(i) = null_performance.pos.r_rank(1);
                   null_pos_r_pearson(i) = null_performance.pos.r_pearson(1);
                   null_neg_r_rank(i) = null_performance.neg.r_rank(1);
                   null_neg_r_pearson(i) = null_performance.neg.r_pearson(1);
                   null_mean_y_predict(i, :) = null_y_predict.mean;
                   null_pos_y_predict(i, :) = null_y_predict.pos;
                   null_neg_y_predict(i, :) = null_y_predict.neg;
               end
               
               % Calculate p-value and store results
               r = test.performance.mean.r_pearson(1);
               count = sum([null_mean_r_pearson] > r);
               fprintf('%s %s %s permutation p, %1.4e \n', pheno, pill, stim, count/n)
               
               % Compile results
               null.mean_r_rank = null_mean_r_rank;
               null.mean_r_pearson = null_mean_r_pearson;
               null.pos_r_rank = null_pos_r_rank;
               null.pos_r_pearson = null_pos_r_pearson;
               null.neg_r_rank = null_neg_r_rank;
               null.neg_r_pearson = null_neg_r_pearson;
               null.mean_y_predict = null_mean_y_predict;
               null.pos_y_predict = null_pos_y_predict;
               null.neg_y_predict = null_neg_y_predict;
               
               % Save results
               save(outputfile, 'null', 'test')
           else
               disp('single run')
           end
       end
   end
end