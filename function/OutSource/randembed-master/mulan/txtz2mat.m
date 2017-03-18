function txtz2mat(stem)
    quiet=struct('InfoLevel',0);    
    xt = txt2mat(sprintf('%s-train.x.txt', stem), quiet);
    xs = txt2mat(sprintf('%s-test.x.txt', stem), quiet);
    yt=sparse(txt2mat(sprintf('%s-train.y.txt', stem), quiet));
    ys=sparse(txt2mat(sprintf('%s-test.y.txt', stem), quiet));
    file=sprintf('%s.mat',stem);
    save(file,'xt','xs','yt','ys','-v7.3')
end

