% A simple demo for locally polynomial regression.

f = @(x) sum(cos(x),2);
dim = 5;
Xtr = rand(100, dim);
Xte = rand(1000, dim);
Ytr = f(Xtr);
Yte = f(Xte);


plotCols = {'b', 'g', 'r', 'k', 'c', 'g', 'y'};
polyOrders = [0, 1, 2, 0, 1, 2];
kernels = {'gauss', 'gauss', 'gauss', 'legendre', 'legendre', 'legendre'};
legKernelOrders = [0, 0, 0, 2, 2, 1];

numExperiments = numel(kernels);
predFuncs = cell(numExperiments, 1);

figure;

for expIter = 1:numExperiments
  kernelParams.kernelType = kernels{expIter};
  kernelParams.order = legKernelOrders(expIter);
  predFunc = localPolyRegressionCV(Xtr, Ytr, [], polyOrders(expIter), kernelParams);
  Ypred = predFunc(Xte);
  err = norm(Ypred - Yte)^2;
  fprintf('Experiment %d: polyOrder: %d, kernel: %s, err: %0.4f.\n', ...
    expIter, polyOrders(expIter), kernels{expIter}, err);
  predFuncs{expIter} = predFuncs;
end

% N.B: the implementation also cross validates for the order of the polynomial but we
% are fixing it for the purpose of this demo.
