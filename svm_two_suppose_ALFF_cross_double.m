% 针对FC脑连接矩阵对称性，进行输出修正  2020-11-17


function [train_h0_data, train_h0_label, train_h1_data, train_h1_label, test_h0_label, test_h1_label, rank_h0, rank_h1]=svm_two_suppose_ALFF(index, train_data_name, test_data_name)
% Include dependencies
warning('off');
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% data_name = 'Peking_1_data';
% index = 4;

listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};

% [ methodID ] = readInput( listFS );
selection_method = listFS{10}; % Selected rfe

% Load the data and select features for classification
% load fisheriris
load([train_data_name '.mat']);
train_alff = alff(:,[21:22, 29:42, 71:78, 91:116]); 
train_tag = tag;

load([test_data_name '.mat']);
test_alff = alff(index,[21:22, 29:42, 71:78, 91:116]);
test_tag = tag(index,:);

switch train_data_name
    case 'NYU_alff_ds'
        test_ind = 217;
    case 'PU_alff_ds'
        test_ind = 195;
    case 'KKI_alff_ds'
        test_ind = 84;
    case 'NI_alff_ds'
        test_ind = 49;
    otherwise 
        test_ind = -1;
end

% X_temp = inform.brain_conn_show;    % FC对称矩阵
X_temp = cat(1, train_alff, test_alff);                 % FC上三角矩阵
% X_temp = alff;
Y_temp = cat(1, train_tag, test_tag);
Y_temp = nominal(ismember(Y_temp,0));
Y_temp = logical(double(Y_temp)-1);
X = X_temp;

%for i=1:max(size(X_temp))    
%   temp = X_temp{1,i}';
%   X = [X; temp];                                    % modified by tyb 2022-4-22
%end

% X = meas; clear meas
% Extract the Setosa class
Y = nominal(ismember(Y_temp,1)); 

[train_h0_data, train_h0_label, test_h0_label, rank_h0] = train_h0(test_ind, X, Y, selection_method, Y_temp);

[train_h0_data, train_h0_label] = energy_normalization(train_h0_data, train_h0_label);

[train_h1_data, train_h1_label, test_h1_label, rank_h1] = train_h1(test_ind, X, Y, selection_method, Y_temp);

[train_h1_data, train_h1_label] = energy_normalization(train_h1_data, train_h1_label);
end

function [train_data_out, train_label_out] = energy_normalization(train_data, train_label)

    tmp = train_data';
    sample_energy_tmp = sqrt(sum(tmp.^2));
    
    agv_energy_1 = mean(sample_energy_tmp(train_label));
    avg_energy_0 = mean(sample_energy_tmp(~train_label));
    sizeoftmp = size(tmp);
    sample_energy_map = ones(1, sizeoftmp(2));
    sample_energy_map(train_label) = agv_energy_1;
    sample_energy_map(~train_label) = avg_energy_0;
    
    % sample_energy_map2 = [agv_energy_1*ones(1,length(sample_energy_tmp(train_label))) avg_energy_0*ones(1,length(sample_energy_tmp(~train_label)))];
    energy_map = ones(size(tmp,1),1) * sample_energy_map;
    
    train_data_out = (tmp ./ energy_map)'; 
    train_label_out = train_label;
    
    return
end

function [train_h0_data, train_h0_label, test_h0_label, rank_h0] = train_h0(index, X, Y, selection_method, Y_temp)

X_train = double(X);
Y_train = (double(Y)-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1; % labels: neg_class -1, pos_class +1
test_h0_label = double(Y(index));

%numF = size(X_train,2);
% numF = 110;   %tyb 2020-9-4
numF = 25;   %tyb 2020-11-17

% feature Selection on training data
    switch lower(selection_method)
        case 'ilfs'
            % Infinite Latent Feature Selection - ICCV 2017
            [ranking, weights, subset] = ILFS_auto(X_train, Y_train , 4, 0 );
        case 'mrmr'
            ranking = mRMR(X_train, Y_train, numF);
        
        case 'relieff'
            [ranking, w] = reliefF( X_train, Y_train, 20);
        
        case 'mutinffs'
            [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
        case 'fsv'
            [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
        case 'laplacian'
            W = dist(X_train');
            W = -W./max(max(W)); % it's a similarity
            [lscores] = LaplacianScore(X_train, W);
            [junk, ranking] = sort(-lscores);
        
        case 'mcfs'
            % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
            options = [];
            options.k = 5; %For unsupervised feature selection, you should tune
            %this parameter k, the default k is 5.
            options.nUseEigenfunction = 4;  %You should tune this parameter.
            [FeaIndex,~] = MCFS_p(X_train,numF,options);
            ranking = FeaIndex{1};
        
        case 'rfe'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'l0'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'fisher'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'inffs'
            % Infinite Feature Selection 2015 updated 2016
            alpha = 0.5;    % default, it should be cross-validated.
            sup = 1;        % Supervised or Not
            [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
        case 'ecfs'
            % Features Selection via Eigenvector Centrality 2016
            alpha = 0.5; % default, it should be cross-validated.
            ranking = ECFS( X_train, Y_train, alpha )  ;
        
        case 'udfs'
            % Regularized Discriminative Feature Selection for Unsupervised Learning
            nClass = 2;
            ranking = UDFS(X_train , nClass ); 
        
        case 'cfs'
            % BASELINE - Sort features according to pairwise correlations
            ranking = cfs(X_train);     
        
        case 'llcfs'   
            % Feature Selection and Kernel Learning for Local Learning-Based Clustering
            ranking = llcfs( X_train );
        
        otherwise
            disp('Unknown method.')
    end

   % k = 110; % select the first 110 features
    k = numF; % select the first 55 features

    %svmStruct = fitcsvm(X_train(:,ranking<=k),Y_train,'Standardize',true,'KernelFunction','RBF',...
    %'KernelScale','auto','OutlierFraction',0.0);

    %C = predict(svmStruct,X_train(:,ranking<=k));
    %err_rate = sum(Y_train~= C)/max(size(Y_train)); % mis-classification rate
    %% conMat = confusionmat(Y_test,C); % the confusion matrix
    %fprintf('\nMethod %s (Linear-SVMs): Accuracy: %.2f%%, Error-Rate: %.2f \n',...
    %    selection_method,100*(1-err_rate),err_rate);
   
    rank_h0 = ranking(1:k);
    train_h0_data = X_train(:,ranking(1:k));
    train_h0_label = Y_temp;

    train_h0_data(index,:)=[];
    train_h0_label(index)=[];

    [~,indx_1] = sort(train_h0_label,'descend');
    train_h0_label= train_h0_label(indx_1);
    train_h0_data = train_h0_data(indx_1,:);

    return
end

function [train_h1_data, train_h1_label, test_h1_label, rank_h1] = train_h1(index, X, Y, selection_method, Y_temp)

X_train = double(X);
Y_temp(index) = ~Y_temp(index);
if Y(index) == 'true'
    Y(index,1) = 'false';
else
    Y(index,1) = 'true';
end

Y_train = (double(Y)-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1; % labels: neg_class -1, pos_class +1
test_h1_label = double(Y(index));

%numF = size(X_train,2);
% numF = 110;
numF = 25;   %tyb 2020-11-17

% feature Selection on training data
    switch lower(selection_method)
        case 'ilfs'
            % Infinite Latent Feature Selection - ICCV 2017
            [ranking, weights, subset] = ILFS_auto(X_train, Y_train , 4, 0 );
        case 'mrmr'
            ranking = mRMR(X_train, Y_train, numF);
        
        case 'relieff'
            [ranking, w] = reliefF( X_train, Y_train, 20);
        
        case 'mutinffs'
            [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
        case 'fsv'
            [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
        case 'laplacian'
            W = dist(X_train');
            W = -W./max(max(W)); % it's a similarity
            [lscores] = LaplacianScore(X_train, W);
            [junk, ranking] = sort(-lscores);
        
        case 'mcfs'
            % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
            options = [];
            options.k = 5; %For unsupervised feature selection, you should tune
            %this parameter k, the default k is 5.
            options.nUseEigenfunction = 4;  %You should tune this parameter.
            [FeaIndex,~] = MCFS_p(X_train,numF,options);
            ranking = FeaIndex{1};
        
        case 'rfe'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'l0'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'fisher'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'inffs'
            % Infinite Feature Selection 2015 updated 2016
            alpha = 0.5;    % default, it should be cross-validated.
            sup = 1;        % Supervised or Not
            [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
        case 'ecfs'
            % Features Selection via Eigenvector Centrality 2016
            alpha = 0.5; % default, it should be cross-validated.
            ranking = ECFS( X_train, Y_train, alpha )  ;
        
        case 'udfs'
            % Regularized Discriminative Feature Selection for Unsupervised Learning
            nClass = 2;
            ranking = UDFS(X_train , nClass ); 
        
        case 'cfs'
            % BASELINE - Sort features according to pairwise correlations
            ranking = cfs(X_train);     
        
        case 'llcfs'   
            % Feature Selection and Kernel Learning for Local Learning-Based Clustering
            ranking = llcfs( X_train );
        
        otherwise
            disp('Unknown method.')
    end

    %k = 110; % select the first 110 features
    k =numF; % select the first 55 features
    
    
    %svmStruct = fitcsvm(X_train(:,ranking<=k),Y_train,'Standardize',true,'KernelFunction','RBF',...
    %'KernelScale','auto','OutlierFraction',0.0);

    %C = predict(svmStruct,X_train(:,ranking<=k));
    %err_rate = sum(Y_train~= C)/max(size(Y_train)); % mis-classification rate
    %% conMat = confusionmat(Y_test,C); % the confusion matrix
    %fprintf('\nMethod %s (Linear-SVMs): Accuracy: %.2f%%, Error-Rate: %.2f \n',...
    %    selection_method,100*(1-err_rate),err_rate);

   
    rank_h1 = ranking(1:k);
    train_h1_data = X_train(:,ranking(1:k));
    train_h1_label = Y_temp;

    train_h1_data(index,:)=[];
    train_h1_label(index)=[];

    [~,indx_1] = sort(train_h1_label,'descend');
    train_h1_label= train_h1_label(indx_1);
    train_h1_data = train_h1_data(indx_1,:);

    return
end
