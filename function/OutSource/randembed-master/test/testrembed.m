clear all;
close all;
more off;

addpath('../matlab/')

randn('seed',8675309);
rand('seed',90210);

n=1009;
d=2007;
hk=105;
c=47;
k=11;

H=randn(n,hk);
Wx=randn(hk,d);
X=H*Wx+0.01*randn(n,d);
Wy=randn(hk,c);
Y=H*Wy+0.01*randn(n,c);

fprintf('calling rembed ...');
embed=rembed(X',ones(1,n),Y',k,struct('tmax',4,'lambda',1e-4));
fprintf(' done\n');

fprintf('performing exact svd ...');
[Ux,Sx,Vx]=svd(X);
[~,Sy,Vy]=svds(Ux'*Y,k);
fprintf(' done\n');

angles=[];
for ii=1:k
  angles=[angles; subspace(embed.Wy(:,1:ii),Vy(:,1:ii))];
end
if max(angles)<0.01
  fprintf('*** test pass ***\n');
else
  error('canonical angles are larger than expected')
  angles
end
