function IoU = IntersectionOverUnion(crack, groudTruth)
    % Get the biary versions of obtained (SVM) crack and ground truth. 
    % IoU = Intersection / Union

    % Calculate the intersection
    intersection = crack & groudTruth;
    % Calculate the union
    union = crack | groudTruth;
    % area calculation
    intersectionArea = nnz(intersection);
    unionArea = nnz(union);                
    % IoU
    IoU = intersectionArea / unionArea;
end