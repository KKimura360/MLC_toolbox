function [] = writeData(data, dname)

train_name = [dname, '_train.txt'];
test_name = [dname, '_test.txt'];

write_data(data.X',data.Y',train_name);
write_data(data.Xt',data.Yt',test_name);

end