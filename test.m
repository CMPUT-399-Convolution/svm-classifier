setup

[Xtr, ytr, Xtest, ytest] = loadMNIST('feat');

multisvm = load('multisvm.mat');

[ preds, maxconfs ] = multisvmpred(multisvm, Xtest);

correct = ytest == preds;

acc = (sum(correct)/length(mask))*100;
disp(['Test accuracy = ',num2str(acc),'%.']);
