function [ multisvm ] = multisvmtrain( X, y, kernel, kerneloption, verbose, gpus)
if nargin < 6
        gpus = [];
    end
    if nargin < 5
        verbose = 0;
    end
    if nargin < 4
        kerneloption = 1;
    end
    if nargin < 3
        kernel = 'poly';
    end

    epsilon = 0.000001;

    tic;
    multisvm = struct;
    multisvm.uniqueLabels = unique(y);
    multisvm.nlabels = length(multisvm.uniqueLabels);
    multisvm.classifiers = struct('W',cell(1,multisvm.nlabels),'b',0,'label',0);

    % setup GPUs
    numGpus = numel(gpus) ;
    if numGpus > 1
        if isempty(gcp('nocreate')),
            parpool('local',numGpus) ;
            spmd, gpuDevice(opts.gpus(labindex)), end
      end
  elseif numGpus == 1
      gpuDevice(gpus)
  end


  if numGpus >= 1
      multisvm = gpuArray(multisvm);
      X = gpuArray(X);
      y = gpuArray(y);
  end

  % initial C value is hardcoded, later we should cross validate for it
  Cs = [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000 ];
  disp('multisvm - starting creation of SVM classifiers');
  for i = 1:multisvm.nlabels;
        class = multisvm.uniqueLabels(i);

        % y - one vs all
        yova = y;
        yova(y==class) = 1;
        yova(y~=class) = -1;

        % cross validate for C
        accbest = -inf;
        for C = Cs;
            [Xsup,yalpha,b,~] = svmclass(X,yova,C,epsilon,kernel,kerneloption,verbose);
            [~,accval,~] = svmvalmod(X,yova,Xsup,yalpha,b,kernel,kerneloption);
            W = (yalpha'*Xsup)';
            if accval > accbest
                accbest = accval;
                Wbest = W;
                bbest = b;
                Cbest = C;
            end
        end
        multisvm.classifiers(i).label = class;
        multisvm.classifiers(i).b = bbest;
        multisvm.classifiers(i).W = Wbest;
        multisvm.classifiers(i).C = Cbest;
        disp(['  SVM classifier created for class="',num2str(class),'" with ',num2str(accval*100,'%1.1f'),'% training accuracy']);
    end
    toc;

  if numGpus >= 1
      multisvm = gather(multisvm);
      X = gather(X);
      y = gather(y);
  end


end
