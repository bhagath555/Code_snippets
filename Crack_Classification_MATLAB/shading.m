function corrected_image = shading(I_gray)
% Applying gaussian filter with large sigma
sigma = 50;  
background = imgaussfilt(I_gray, sigma);

I_gray = double(I_gray);
background = double(background);

% Normalize
corrected_image = I_gray ./ background;

% changeing to 0 - 255 range
corrected_image = uint8(255 * mat2gray(corrected_image));
end

