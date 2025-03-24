% Directory paths
inputDir = 'training/';   % Input directory of trained images
outputDir = 'training_morph/'; % Output directory for morphed images

% This funciton applys morphological operations to all image in input dir.
% And, stores them in outputDir
morph_ops(inputDir, outputDir);

%% Applying morphological operators to testing images
% Directory paths
inputDir = 'testing/';   % Input directory of trained images
outputDir = 'testing_morph/'; % Output directory for morphed images
% This funciton applys morphological operations to all image in input dir.
% And, stores them in outputDir
morph_ops(inputDir, outputDir);


%% morph_ops Implementation

function morph_ops(input_dir, output_dir)
    % All the images in the directory
    imageFiles = dir(fullfile(input_dir, '*.jpg'));
    
    % Check if output directory exists, if not, create it
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Loop through each file in the directory
    for k = 1:length(imageFiles)
    
        % Get the file name
        fileName = imageFiles(k).name;
        inputFilePath = fullfile(input_dir, fileName);
        
        % Read the color image
        color_image = imread(inputFilePath);
        
        % Convert to grayscale
        gray_image = rgb2gray(color_image);
        
        % Apply shading
        gray_image = shading(gray_image);
        
        % Adjust contrast
        gray_image = imadjust(gray_image, stretchlim(gray_image), []);
        
        % Binarize the image with adaptive thresholding
        bin = ~imbinarize(gray_image, 'adaptive', 'Sensitivity', 1);
        
        % Morphological operations
        se = strel('square', 2);
        bin_morph = imopen(bin, se);
        bin_morph = bwmorph(bin_morph, 'bridge');
        bin_morph = bwmorph(bin_morph, 'fill');
        bin_morph = imclose(bin_morph, se);
        
        % Remove small objects based on area
        mx = round(0.05 * max([regionprops(bin, 'Area').Area]));
        bin_filter = bwareaopen(bin_morph, mx);
        
        % Final morphological operation
        final_image = bwmorph(bin_filter, 'diag');
        
        final_image_uint8 = 255* uint8(final_image); 
        
        % Create the output file path
        outputFilePath = fullfile(output_dir, [fileName(1:end-4), '.png']);
        
        % Save the final image, preserving dimensions and data
        imwrite(final_image_uint8, outputFilePath, 'png'); 
    end

end







