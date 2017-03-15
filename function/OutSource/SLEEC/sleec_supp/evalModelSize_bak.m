%SVPModel contains trEmbed and alpha
%clusterCenters{} contains the cluster centers

mSize = zeros(T, 1);
curModelSize = 0;
wSize = 0;
CCsize = 0;
embedSize = 0;

for i = 1:T
    assign = assign_mat(i, :);
    membed = SVPModel{i}.trEmbed;
    sp_membed = {};
    for j = 1:max(assign)+1
        sp_membed{j} = sparse(membed{j});
    end
        
    mW = SVPModel{i}.alpha;
    mCC = clusterCenters{i};
    tmp = whos('sp_membed');
    curModelSize= curModelSize + (tmp.bytes/1024^3)/2;
    embedSize = embedSize + (tmp.bytes/1024^3)/2;
    tmp = whos('mCC');
    curModelSize= curModelSize + (tmp.bytes/1024^3)/2;
    CCsize = CCsize + (tmp.bytes/1024^3)/2;
    tmp = whos('mW');
    curModelSize= curModelSize + (tmp.bytes/1024^3)/2;
    wSize = wSize + (tmp.bytes/1024^3)/2;
    mSize(i) = curModelSize;
end