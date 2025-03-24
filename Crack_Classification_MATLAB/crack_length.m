function length = crack_length(img)
    % Skeletonize the image
    sk_img = bwskel(img);
    % Calculate the non zero pixels = length of the crack in pixels.
    length = nnz(sk_img);
end