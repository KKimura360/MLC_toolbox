function [U, V]=WAltMin_asymm(Om, OmVal, mxitr, tol, Uinit, Vinit, numThreads)
U=Uinit;
V=Vinit;
oerr=+Inf;
fact=0.3;
nMOmega=norm(OmVal,'fro');
m = size(U, 1);
n = size(V, 1);
U_old=U;
V_old=V;
U_old1 = U;
V_old1 = V;

[ind_omega3, OmIdx] = sort(Om);
S1 = OmVal(OmIdx);
[I1, J1] = ind2sub([m n], ind_omega3);

[I1v, I1i] = sort(I1);
J2 = J1(I1i);
S2 = S1(I1i);
ind_omega2 = sub2ind([m n], I1v, J2);

histJ = hist(I1v, m);
histJ = cumsum(histJ);
histJ = [0, histJ];

histI = hist(J1, n);
histI = cumsum(histI);
histI = [0, histI];

oerr_st = oerr;
fprintf('Iteration, Err =            ');

%keyboard
for itr=1:mxitr
    
    uv_omega=compute_X_Omega(U, V, ind_omega3, numThreads);
    err=norm(uv_omega-S1,'fro')/sqrt(length(uv_omega));
    
    fprintf('\b\b\b\b\b\b\b\b\b\b\b%4d %1.4f', itr, err);
    
    if(err<tol)
        break;
    end
    if(oerr-err<tol*.1)
        if(oerr < err)
            U = U_old;
            V = V_old;
            fprintf('\n');
            break;
        else
            fprintf('\n');
            break;
        end
    end
    
    eta=1/(fact*nMOmega);
    U_old=U;
    V_old=V;
    oerr_st = err;
    [U, ~] = updateU(U, V, ind_omega2, S2, eta, histJ, numThreads);
    [V, ~] = updateV(U, V, ind_omega3, S1, eta, histI, numThreads);
    oerr=err;
end
fprintf('\n');