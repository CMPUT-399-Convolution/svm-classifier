function [ Xtrain, ytrain, Xtest, ytest ] = loadMNIST ( type )
    trainfile = ['mnist/mnist-train-',type,'.mat'];
    testfile = ['mnist/mnist-test-',type,'.mat'];
    
    % load all MNIST training data
    try
        load(trainfile)
    catch 
        [Xtrain, ytrain] = loadMNISTPair('./mnist', 'train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
        if strcmp(type,'feat') == 0
            Xtrain = convert2hog(Xtrain,9);
        end
        save(trainfile, 'Xtrain', 'ytrain');
    end
    disp('MNIST Training image features loaded.');
    
    % load all MNIST testing data
    try
        load(testfile)
    catch
        [Xtest, ytest] = loadMNISTPair('./mnist', 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
         if strcmp(type,'feat') == 0
            Xtest = convert2hog(Xtest,9);
         end
        save(testfile, 'Xtest', 'ytest');
    end
    disp('MNIST Testing image features loaded.');
end

function [ X, y ] = loadMNISTPair( directory, image_file, label_file )
    X = loadMNISTImages([directory,'/',image_file]);
    X = X';
    y = loadMNISTLabels([directory,'/',label_file]);
end

function [ Xfeat ] = convert2hog ( X, cell_size )
    [hsz, wsz] = size(X);
    disp(['Calculating hog features for ',num2str(hsz),' images.']);
    for i = 1:hsz
        I = single(reshape(X(i,:),[28 28]));
        F = vl_hog(I, cell_size);
        if not(exist('Xfeat', 'var'))
            Xfeat = zeros(wsz, numel(F));
        end
        Xfeat(i,:) = reshape(F, [1 numel(F)]);
    end
end