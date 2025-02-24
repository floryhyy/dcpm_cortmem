%% cpm per run for item memory and arousal

% Define analysis parameters
pheno_ls = ["RecogAcc_coded" "Arous_All"];
stim_ls = ["alc" "tool"];
pill_ls = ["cortisol" "placebo"];
p = 0.01;
folder = "../results/dynamic_connectivity/";

% Initialize result storage arrays
behavior_col = [];
stim_col = [];
pill_col = [];
mean_rho = [];
pos_net_size = [];
neg_net_size = [];

for pheno = pheno_ls
   for stim = stim_ls
       for pill = pill_ls
           % Load connectome data
           filename = append(folder, 'n26_', pheno, '_', pill, '_', stim, '_trialID_dynamic_connectome.mat');
           data = load(filename);
           x = [data.X];
           x = permute(x, [2 3 1]);
           y = [data.y]';
           id = [data.id];
           
           % Run appropriate CPM based on phenotype
           if pheno == "RecogAcc_coded"
               categorical = false;
               [y_predict, performance, final_net, mean_net] = cpm_singlefunc_log(x, y, 'kfolds', 40, 'id', id, 'categorical', categorical, 'flatten', false, 'p', p);
           else
               [y_predict, performance, final_net, mean_net] = cpm_singlefunc(x, y, 'kfolds', 40, 'id', id, 'flatten', false, 'p', p);
           end
           
           % Store results for summary table
           behavior_col = [behavior_col; pheno];
           stim_col = [stim_col; stim];
           pill_col = [pill_col; pill];
           mean_rho = [mean_rho; performance.mean.r_pearson(1)];
           pos_net_size = [pos_net_size; sum(sum(final_net.pos))/2];
           neg_net_size = [neg_net_size; sum(sum(final_net.neg))/2];
           
           % Save network edges
           pos_net = append('result/network_edges/n26_', pheno, '_byTrial_', pill, '_', stim, '_htBPfiltered_p0.01_pos.txt');
           neg_net = append('result/network_edges/n26_', pheno, '_byTrial_', pill, '_', stim, '_htBPfiltered_p0.01_neg.txt');
           writematrix(final_net.pos, pos_net, 'Delimiter', ',')
           writematrix(final_net.neg, neg_net, 'Delimiter', ',')
       end
   end
end

% Create and save summary table
summary_table = table(behavior_col, stim_col, pill_col, mean_rho, pos_net_size, neg_net_size);
writetable(summary_table, 'result/all_cpm_results.csv')


%% run code - predicting across construct
% Initialize parameters
stim_ls = ["alc" "tool"];
pill_ls = ["cortisol" "placebo"];
folder = "../results/dynamic_connectivity/";
p = 0.01;
pheno_ls = ["RecogAcc_coded" "Arous_All"];

% Initialize results storage
train_col = [];
test_col = [];
stim_col = [];
pill_col = [];
mean_rho = [];
pos_net_size = [];
neg_net_size = [];

for pheno = pheno_ls
    for stim = stim_ls
        for pill = pill_ls
            % Set up training and testing phenotypes
            if pheno == "RecogAcc_coded"
                filename = append(folder, 'n26_Arous_All_', pill, '_', stim, '_trialID_dynamic_connectome.mat');
                train_col = [train_col; "Arous_All"];
                test_col = [test_col; "RecogAcc_coded"];
            else
                filename = append(folder, 'n26_RecogAcc_coded_', pill, '_', stim, '_trialID_dynamic_connectome.mat');
                train_col = [train_col; "RecogAcc_coded"];
                test_col = [test_col; "Arous_All"];
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
            
            % Set prediction type
            if pheno == "RecogAcc_coded"
                predicted_type = "binary";
            else
                predicted_type = "linear";
            end
            
            % Run CPM
            [y_predict, performance, final_net, mean_net] = cpm_diff_corr_train(x, x_model, testX, y, y_model, testY, 'kfolds', 40, 'id', id, 'train_id', train_id, 'test_id', test_id, 'predicted_type', predicted_type, 'flatten', false, 'pthresh', p);
            
            % Store results
            stim_col = [stim_col; stim];
            pill_col = [pill_col; pill];
            mean_rho = [mean_rho; performance.mean.r_pearson(1)];
            pos_net_size = [pos_net_size; sum(sum(final_net.pos))/2];
            neg_net_size = [neg_net_size; sum(sum(final_net.neg))/2];
        end
    end
end

% Create and save summary table
summary_table = table(train_col, test_col, stim_col, pill_col, mean_rho, pos_net_size, neg_net_size);
summary_table.Properties.VariableNames = ["train on", "test on", "stim", "pill", "rho", "pos network", "neg network"];
writetable(summary_table, 'result/generalization_across_construct_cpm_results.csv');

%% Run code - predicting same construct across runs (stimulus generalization)
% This script uses CPM to predict behavioral metrics across different stimulus types
% while keeping pill condition and phenotype consistent.

% Define analysis parameters
pheno_ls = ["RecogAcc_coded" "Arous_All"];
pill_ls = ["cortisol" "placebo"];
stim_ls = ["alc" "tool"];
folder = "../results/dynamic_connectivity/";
p = 0.01;

% Initialize result storage arrays
behavior_col = [];
pill_col = [];
mean_rho = [];
temp_p = [];
pos_net_size = [];
neg_net_size = [];
train_col = [];
test_col = [];

for pheno = pheno_ls
   for stim = stim_ls
       for pill = pill_ls
           % Define training set based on stimulus type
           if stim == "alc"
               filename = append(folder, 'n26_', pheno, '_', pill, '_tool_trialID_dynamic_connectome.mat');
               train_col = [train_col; "tool"];
               test_col = [test_col; "alc"];
           else
               filename = append(folder, 'n26_', pheno, '_', pill, '_alc_trialID_dynamic_connectome.mat');
               train_col = [train_col; "alc"];
               test_col = [test_col; "tool"];
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
           
           % Set prediction type based on phenotype
           if pheno == "RecogAcc_coded"
               predicted_type = "binary";
           else
               predicted_type = "linear";
           end
           
           % Run CPM with cross-stimulus training and testing
           [y_predict, performance, final_net, mean_net] = cpm_diff_corr_train(x, x_model, testX, y, y_model, testY, 'kfolds', 40, 'id', id, 'train_id', train_id, 'test_id', test_id, 'predicted_type', predicted_type, 'flatten', false, 'pthresh', p);
           
           % Store results
           behavior_col = [behavior_col; pheno];
           pill_col = [pill_col; pill];
           mean_rho = [mean_rho; performance.mean.r_pearson(1)];
           temp_p = [temp_p; performance.mean.r_pearson(2)];
           pos_net_size = [pos_net_size; sum(sum(final_net.pos))/2];
           neg_net_size = [neg_net_size; sum(sum(final_net.neg))/2];
       end
   end
end

% Create and save summary table
summary_table = table(train_col, test_col, pill_col, behavior_col, mean_rho, temp_p, pos_net_size, neg_net_size);
summary_table.Properties.VariableNames = ["train on", "test on", "pill", "behavior", "rho", "temp_p", "pos network", "neg network"];
writetable(summary_table, 'result/generalization_across_stim_cpm_results.csv');

%% Calculate permutation p-values and augment result tables
% This script loads permutation results and adds p-values to three analysis tables:
% 1. Direct phenotype prediction
% 2. Cross-phenotype generalization 
% 3. Cross-stimulus generalization

% Define analysis parameters
pheno_ls = ["RecogAcc_coded" "Arous_All"];
stim_ls = ["alc" "tool"];
pill_ls = ["cortisol" "placebo"];
n_permutations = 1000;

% 1. Direct phenotype prediction results
result_table = readtable('result/all_cpm_results.csv');
result_table.Properties.VariableNames = ["pheno", "stim", "pill", "mean_rho", "pos network", "neg network"];

% Initialize result vectors
[mean_pvalue, pos_rho, pos_pvalue, neg_rho, neg_pvalue] = deal([]);

% Calculate permutation p-values for each condition
for pheno = pheno_ls
   for stim = stim_ls
       for pill = pill_ls
           % Determine file path based on phenotype
           if pheno == "Arous_All"
               filename = append('result/permutation/linear_', pheno, '_byTrial_', pill, '_', stim, '_p0.01.mat');
           else
               filename = append('result/permutation/', pheno, '_byTrial_', pill, '_', stim, '_p0.01.mat');
           end
           
           % Load permutation results and compute p-values
           load(filename);
           p_mean = sum(test.performance.mean.r_pearson(1) < null.mean_r_pearson)/n_permutations;
           mean_pvalue = [mean_pvalue; p_mean];
           
           r_pos = test.performance.pos.r_pearson(1);
           p_pos = sum(r_pos < null.pos_r_pearson)/n_permutations;
           pos_rho = [pos_rho; r_pos];
           pos_pvalue = [pos_pvalue; p_pos];
           
           r_neg = test.performance.neg.r_pearson(1);
           p_neg = sum(r_neg < null.neg_r_pearson)/n_permutations;
           neg_rho = [neg_rho; r_neg];
           neg_pvalue = [neg_pvalue; p_neg];
           
           fprintf('%s %s %s: permutation pvalue = %f\n', pheno, pill, stim, p_neg);
       end
   end
end

% Add p-values to result table and save
result_table = addvars(result_table, mean_pvalue, pos_rho, pos_pvalue, neg_rho, neg_pvalue, 'After', "mean_rho");
writetable(result_table, 'result/all_cpm_results_with_pvalues.csv');

% 2. Cross-phenotype generalization results
result_table = readtable('result/generalization_across_construct_cpm_results.csv');
result_table.Properties.VariableNames = ["train on", "test on", "stim", "pill", "mean_rho", "pos network", "neg network"];

% Reset result vectors
[mean_pvalue, pos_rho, pos_pvalue, neg_rho, neg_pvalue] = deal([]);

% Calculate p-values for cross-phenotype generalization
for pheno = pheno_ls
   for stim = stim_ls
       for pill = pill_ls
           % Determine file path based on test phenotype
           if pheno == "RecogAcc_coded"
               filename = append('result/permutation/Arous_All_RecogAcc_coded_byTrial_', pill, '_', stim, '_p0.01.mat');
           else
               filename = append('result/permutation/linear_RecogAcc_coded_Arous_All_byTrial_', pill, '_', stim, '_p0.01.mat');
           end
           
           % Load permutation results and compute p-values
           load(filename);
           p_mean = sum(test.performance.mean.r_pearson(1) < null.mean_r_pearson)/n_permutations;
           mean_pvalue = [mean_pvalue; p_mean];
           
           r_pos = test.performance.pos.r_pearson(1);
           p_pos = sum(r_pos < null.pos_r_pearson)/n_permutations;
           pos_rho = [pos_rho; r_pos];
           pos_pvalue = [pos_pvalue; p_pos];
           
           r_neg = test.performance.neg.r_pearson(1);
           p_neg = sum(r_neg < null.neg_r_pearson)/n_permutations;
           neg_rho = [neg_rho; r_neg];
           neg_pvalue = [neg_pvalue; p_neg];
           
           fprintf('%s %s %s: permutation pvalue = %f\n', pheno, pill, stim, p_neg);
       end
   end
end

% Add p-values to result table and save
result_table = addvars(result_table, mean_pvalue, pos_rho, pos_pvalue, neg_rho, neg_pvalue, 'After', "mean_rho");
writetable(result_table, 'result/generalization_across_construct_cpm_results_with_pvalues.csv');

% 3. Cross-stimulus generalization results
result_table = readtable('result/generalization_across_stim_cpm_results.csv');
result_table.Properties.VariableNames = ["train on", "test on", "pill", "behavior", "mean_rho", "temp_p", "pos network", "neg network"];

% Reset result vectors
[mean_pvalue, pos_rho, pos_pvalue, neg_rho, neg_pvalue] = deal([]);

% Calculate p-values for cross-stimulus generalization
for pheno = pheno_ls
   for stim = stim_ls
       for pill = pill_ls
           % Determine file path based on test stimulus and phenotype
           if stim == "alc"
               if pheno == "Arous_All"
                   filename = append('result/permutation/linear_', pheno, '_byTrial_', pill, '_tool_alc_p0.01.mat');
               else
                   filename = append('result/permutation/', pheno, '_byTrial_', pill, '_tool_alc_p0.01.mat');
               end
           else
               if pheno == "Arous_All"
                   filename = append('result/permutation/linear_', pheno, '_byTrial_', pill, '_alc_tool_p0.01.mat');
               else
                   filename = append('result/permutation/', pheno, '_byTrial_', pill, '_alc_tool_p0.01.mat');
               end
           end
           
           % Load permutation results and compute p-values
           load(filename);
           p_mean = sum(test.performance.mean.r_pearson(1) < null.mean_r_pearson)/n_permutations;
           mean_pvalue = [mean_pvalue; p_mean];
           
           r_pos = test.performance.pos.r_pearson(1);
           p_pos = sum(r_pos < null.pos_r_pearson)/n_permutations;
           pos_rho = [pos_rho; r_pos];
           pos_pvalue = [pos_pvalue; p_pos];
           
           r_neg = test.performance.neg.r_pearson(1);
           p_neg = sum(r_neg < null.neg_r_pearson)/n_permutations;
           neg_rho = [neg_rho; r_neg];
           neg_pvalue = [neg_pvalue; p_neg];
           
           fprintf('%s %s %s: permutation pvalue = %f\n', pheno, pill, stim, p_neg);
       end
   end
end

% Add p-values to result table and save
result_table = addvars(result_table, mean_pvalue, pos_rho, pos_pvalue, neg_rho, neg_pvalue, 'After', "mean_rho");
writetable(result_table, 'result/generalization_across_stim_cpm_results_with_pvalues.csv');