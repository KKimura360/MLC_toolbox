function [W, alpha, mu] = computeW(X, Z, rho, lambda1, mxitr, c)
    [n, d] = size(X);
    [~, k] = size(Z);
    alpha = zeros(n, k);
    mu = zeros(n, k);
    
    ph = sqrt(1+rho);
    W = zeros(d, k);
    Xp = X/ph;
    libArgs = ['-s 12 -p 0 -c ', num2str(c), ' -q'];
    for j = 1:mxitr
        Z_temp = (Z+rho*(alpha-mu))/ph;
        for cgs_cnt = 1:k
            model = train(Z_temp(:, cgs_cnt), Xp, libArgs); 
            W(:, cgs_cnt) = model.w;
        end
        alpha = X*W+mu;
        alpha = sign(alpha).*(max(abs(alpha)-(lambda1/rho), 0));
        mu = mu + X*W - alpha;
    end

end