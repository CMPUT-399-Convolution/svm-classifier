% given a matrix of vector row images jitter the images by shifting numPixels
%   -> no additional images are created, images jittered in place
%   -> one direction per each image
function [ X ] = jitter ( X, numPixels ) 
    [hsz, wsz] = size(X);
    jitterAll = [numPixels 0; 0 numPixels; -numPixels 0; 0 -numPixels;numPixels numPixels; -numPixels -numPixels];
    for i = 1:hsz
        jitterInd = mod(i,6) + 1;
        jitter = jitterAll(jitterInd,:);
        I = reshape(X(i,:), [28 28]);
        X(i,:) = reshape(circshift(I,jitter), [1 784]);
    end
end
