% Implementing SVM to classify crack and non-crack regions

% Make sure there are images in 
% training_morph/crack & training_morph/non_crack
% testing_morph

% Read and process images from the 'cracked' directory
TrainingDir = 'training_morph/';
GroundTruthDir = 'ground_truth/gt_training';

% Training images list
TrainingImgs = dir(fullfile(TrainingDir, '*.png'));

% Ground truth images list
GroundTruthImgs = dir(fullfile(GroundTruthDir, '*.png'));

features = [];
labels = [];

if length(TrainingImgs) ~= length(GroundTruthImgs)
    disp('Ground truth images count should match with training images count')
else
    % Loop through each file in the directory
    for k = 1:length(TrainingImgs)
        % Get the file name
        T_fname = TrainingImgs(k).name;
        GT_fname = GroundTruthImgs(k).name;
        % File fullpath
        T_fname = fullfile(TrainingDir, T_fname);
        GT_fname = fullfile(GroundTruthDir, GT_fname);
        
        % Read the binary image
        T_img = imread(T_fname);
        GT_img = imread(GT_fname);

        T_img = imbinarize(T_img(:,:,1));
        GT_img = imbinarize(GT_img(:,:,1));
        
        crack = T_img & GT_img;
        non_crack = T_img & ~ GT_img;

        % figure;
        % subplot(1,3,1); imshow(T_img);
        % subplot(1,3,2); imshow(crack);
        % subplot(1,3,3); imshow(non_crack);

        c = 6;

        % Connected components of crack and non_crack regions
        crack_cc = bwconncomp(crack, 8);
        non_crack_cc = bwconncomp(non_crack, 8);

        % Get region properties
        cfs = gen_feat_labels(crack_cc);
        ncfs = gen_feat_labels(non_crack_cc);

        features = [features; cfs; ncfs];
        labels = [labels; ones(size(cfs, 1), 1); zeros(size(ncfs, 1), 1)];

    end
end

% Now we got the features and labels. Lets feed them into the SVM
SVMModel = fitcsvm(features, labels, 'KernelFunction', 'linear', 'Standardize', true);

% Read the trainig image features 
testingDir = 'testing_morph/';
testingImgs = dir(fullfile(testingDir, '*.png'));
reslutsDir = 'results/';
testingTruDir = 'ground_truth/';
testingTruImgs = dir(fullfile(testingTruDir, '*.png'));
for k= 1:length(testingImgs)
    % Get the file name
    T_fname = testingImgs(k).name;
    % File fullpath
    T_fname = fullfile(testingDir, T_fname);

    % Read the binary image
    T_img = imread(T_fname);
    T_img = imbinarize(T_img(:,:,1));

    % Connected components of crack and non_crack regions
    region_cc = bwconncomp(T_img, 8);
    % Get region properties
    region_fs = gen_feat_labels(region_cc);

    %% Predict labels using the SVM model
    predictedLabels = predict(SVMModel, region_fs);
    % Label each region with different number
    labeledImage = labelmatrix(region_cc); 
    Output = zeros(size(labeledImage));  

    % Color the regions based on predictions
    for regionIdx = 1:region_cc.NumObjects
        if predictedLabels(regionIdx) == 1
            % Crack regions: white
            Output(labeledImage == regionIdx) = 1; 
        else
            % Non-crack regions: black
            Output(labeledImage == regionIdx) = 0;  
        end
    end

    Output = imbinarize(Output);
    sk_img = bwskel(Output);
    % Calculate the non zero pixels = length of the crack in pixels.
    c_l = nnz(sk_img);

    % Testing ground truth (TGT) images
    TGT_fname = testingTruImgs(k).name;
    % File fullpath
    TGT_fname = fullfile(testingTruDir, TGT_fname);
    TGT_img = imread(TGT_fname);
    TGT_img = imbinarize(TGT_img(:,:,1));
    IoU = IntersectionOverUnion(Output, TGT_img);

    % Display the image
    figure;
    subplot(1,3,1); imshow(Output); title('Prediction');
    subplot(1,3,2); imshow(TGT_img); title('Ground truth');
    subplot(1,3,3); imshow(sk_img); title('Skeletonization');

    disp(['Crack length                 : ', num2str(c_l)]);
    disp(['Intersection over Union (IoU): ', num2str(IoU)]);

end

    
function features = gen_feat_labels(regions)
    regions_stats = regionprops(regions, 'Extent', 'Solidity', 'Eccentricity', 'MinorAxisLength', 'MajorAxisLength');

    % aspect_ratio = [regions_stats.MajorAxisLength] ./ [regions_stats.MinorAxisLength];
    features = ...
        [ [regions_stats.Eccentricity];
          [regions_stats.Extent];
          [regions_stats.Solidity];
          % aspect_ratio;
        ]';
end