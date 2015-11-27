function [ preds, maxconfs ] = multisvmpred( multisvm, X )
    hsz = size(X,1);
    
    confs = zeros(hsz,multisvm.nlabels);
    for i=1:multisvm.nlabels
        W = multisvm.classifiers(i).W;
        b = multisvm.classifiers(i).b;
        confs(:,i) = X * W + b;
    end
    
    [maxconfs,inds] = max(confs,[],2);
    preds = multisvm.uniqueLabels(inds); 
end
