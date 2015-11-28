setup

[Xtr, ytr, ~, ~] = loadMNIST('feat');

multisvm = multisvmtrain(Xtr(1:10000,:), ytr(1:10000));

save('multisvm.mat','-struct','multisvm');
