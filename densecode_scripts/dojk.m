arg_list = argv();
str = arg_list{1};

format short g

res = [];

obs = {'chi1B', 'chi2B', 'chi3B', 'chi4B', 'chi1Q', 'chi1Qi', 'cc', 'cci'};
for i = [1:8]
   str0 = strrep(str, 'chi1B', obs{i})
   strres = strrep(str, 'chi1B', 'res');

   data0re = load(str0)(:,1);
   data0im = load(str0)(:,2);

   display('');
   avg0_re = mean(data0re)
   err0_re = std(data0re,1)*sqrt(numel(data0re)-1)

   display('');
   avg0_im = mean(data0im)
   err0_im = std(data0im,1)*sqrt(numel(data0im)-1)

   display('');

   res = [res, avg0_re, err0_re];
   res = [res, avg0_im, err0_im];

end

res = res
dlmwrite(strres, res, " ", "precision", 16)
