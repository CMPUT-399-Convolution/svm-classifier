function [ypred,acc,conf]=svmvalmod(x,y,xsup,yalpha,b,kernel,kerneloption);

% [ypred,acc,conf]=svmvalmod(x,y,xsup,yalpha,b,kernel,kerneloption);
%
% Warper function for svmval, generates predicted labels, accuracy
% and confidence score of SVM classification. See HELP SVMVAL for
% more details.
%

conf = svmval(x,xsup,yalpha,b,kernel,kerneloption);
%ypred = 2*double((conf-b)>0)-1;
ypred = 2*double(conf>0)-1;
acc = sum(y==ypred)/length(y);