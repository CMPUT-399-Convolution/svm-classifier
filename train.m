setup

[Xtr, ytr, Xtest, ytest] = loadMNIST('feat');

multisvm = multisvmtrain(Xtr(1:5000,:), ytr(1:5000));

save('multisvm.mat','-struct','multisvm');
