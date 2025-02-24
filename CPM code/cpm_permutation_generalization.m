%% Code for generating null distribution for cross-phenotype generalization - this file runs on the cluster
% This script runs permutation testing for CPM models that predict across different phenotypes
% (Arousal vs Recognition) while keeping stimulus and pill condition constant.

% Initialize parallel pool
poolobj = parpool;
fprintf('Number of workers: %g\n', poolobj.NumWorkers);

% Define analysis parameters
pheno_ls = ["Arous_All" "RecogAcc_coded"];
stim_ls = ["alc" "tool"];
pill_ls = ["cortisol" "placebo"];
folder = "data/";
p = 0.01;
permutation = true;
n = 1000;  % number of permutations

for pheno = pheno_ls
   for stim = stim_ls
       for pill = pill_ls
           % Define training and testing phenotypes and set output file
           if pheno == "RecogAcc_coded"
               filename = append(folder, 'n26_Arous_All_', pill, '_', stim, '_trialID_dynamic_connectome.mat');
               train_pheno = "Arous_All";
               test_pheno = "RecogAcc_coded";
               predicted_type = "binary";
               outputfile = append('result/permutation/5s_onward_BPfiltered/', train_pheno, '_', test_pheno, '_byTrial_', pill, '_', stim, '_p0.01.mat');
           else
               filename = append(folder, 'n26_RecogAcc_coded_', pill, '_', stim, '_trialID_dynamic_connectome.mat');
               train_pheno = "RecogAcc_coded";
               test_pheno = "Arous_All";
               predicted_type = "linear";
               outputfile = append('result/permutation/5s_onward_BPfiltered/linear_', train_pheno, '_', test_pheno, '_byTrial_', pill, '_', stim, '_p0.01.mat');
           end
           
           % Load training data
           data = load(filename);
           x = data.X;
           x = permute(x, [2 3 1]);
           y = data.y';
           id = data.id;
           
           % Load testing data
           filename = append(folder, 'n26_', pheno, '_', pill, '_', stim, '_trialID_dynamic_connectome.mat');
           data = load(filename);
           x_model = data.X;
           x_model = permute(x_model, [2 3 1]);
           testX = x_model;
           y_model = data.y';
           testY = data.y';
           train_id = data.id;
           test_id = data.id;
           
           % Run actual model
           [y_predict, performance, final_net, mean_net] = cpm_diff_corr_train(x, x_model, testX, y, y_model, testY, 'kfolds', 40, 'id', id, 'train_id', train_id, 'test_id', test_id, 'predicted_type', predicted_type, 'flatten', false, 'pthresh', p);
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
               null_mean_y_predict = zeros(n, size(y_model, 1));
               null_pos_y_predict = zeros(n, size(y_model, 1));
               null_neg_y_predict = zeros(n, size(y_model, 1));
               
               hold on;
               parfor i = 1:n
                   % Shuffle labels
                   shuffled_y = y_model(randperm(size(y_model, 1)));
                   shuffled_y1 = y(randperm(size(y, 1)));
                   
                   % Run CPM on shuffled data
                   [null_y_predict, null_performance, ~, ~] = cpm_diff_corr_train(x, x_model, testX, shuffled_y1, shuffled_y, shuffled_y, 'kfolds', 40, 'id', id, 'train_id', train_id, 'test_id', test_id, 'predicted_type', predicted_type, 'flatten', false, 'pthresh', p);
                   
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