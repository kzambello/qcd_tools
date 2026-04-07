% Copyright 2023 Kevin Zambello <kevin.zambello@pr.infn.it>
%
% This program is free software: you can redistribute it and/or modify it under the terms of
% the GNU General Public License as published by the Free Software Foundation, either
% version 3 of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program.
% If not, see <https://www.gnu.org/licenses/>.

function [res_avg, res_err] = analyze1conf(str, nruns, srcs, params)

   USE_BIASED_PRODUCT = 0;
   TEST_ITERATIVE_FORMULAS = 0;

   Ns = params(1);
   Nt = params(2);
   Nc = 3;
   T = params(3);

   obs_ids = [1,2,3,4,5,6,11,12,13,14,16,17,18,61,62,63,311,312];

   nsrcs = numel(srcs);

   % load data
   data = dlmread(strrep(str, "_runX", ""));
   for i = [2:nruns]
      newdata = dlmread(strrep(str, "X", num2str(i)));
      data = [data; newdata];
   end

   masses = unique(data(:,3));
   ml = min(masses);
   ms = max(masses);

   obs_l = [];
   obs_s = [];

   for i = 1:312

      if isempty(find(obs_ids == i))

          col = zeros(nsrcs, 1);
          obs_l = [obs_l, col];
          obs_s = [obs_s, col];

     else

         idx_l = find(data(:,3) == ml & data(:,1) == i);
         col = data(idx_l, 4) - 1.0j * data(idx_l, 5);
         col = col * Nc*Nt*(Ns^3);
         obs_l = [obs_l, col(srcs,:)];

         idx_s = find(data(:,3) == ms & data(:,1) == i);
         col = data(idx_s, 4) - 1.0j * data(idx_s, 5);
         col = col * Nc*Nt*(Ns^3);
         obs_s = [obs_s, col(srcs,:)];

      end

   end

   % compute avgs

   function [myavg, myerr] = ubmean2(obs1, obs2)
      foo = [];
      for i = [1:nsrcs]
         for j = [1:nsrcs]
            if (i != j)
               foo = [foo; obs1(i)*obs2(j)];
            end
         end
      end
      myavg = mean(foo);
      myerr = 0.0;
   end

   function [myavg, myerr] = ubmean3(obs1, obs2, obs3)
      foo = [];
      for i = [1:nsrcs]
         for j = [1:nsrcs]
            for h = [1:nsrcs]
               if (i != j && i != h && j != h)
                        foo = [foo; obs1(i)*obs2(j)*obs3(h)];
               end
            end
         end
      end
      myavg = mean(foo);
      myerr = 0.0;
   end

   function [myavg, myerr] = ubmean4(obs1, obs2, obs3, obs4)
      foo = [];
      for i = [1:nsrcs]
         for j = [1:nsrcs]
            for h = [1:nsrcs]
               for k = [1:nsrcs]
                  if (i != j && i != h && i != k && j != h && j != k && h != k)
                     foo = [foo; obs1(i)*obs2(j)*obs3(h)*obs4(k)];
                  end
               end
            end
         end
      end
      myavg = mean(foo);
      myerr = 0.0;
   end

   function [myavg, myerr] = ubmeanAA(obsA)
      myavg = (1.0/(nsrcs*(nsrcs-1))) * (sum(obsA)^2 - sum(obsA.^2));
      myerr = 0.0;
   end

   function [myavg, myerr] = ubmeanAAA(obsA)
      myavg = (1.0/(nsrcs*(nsrcs-1)*(nsrcs-2))) * (sum(obsA)^3 - 3.0*sum(obsA)*sum(obsA.^2) + 2.0*sum(obsA.^3) );
      myerr = 0.0;
   end

   function [myavg, myerr] = ubmeanAB(obsA, obsB)
      myavg = (1.0/(nsrcs*(nsrcs-1))) * (sum(obsA)*sum(obsB) - sum(obsA.*obsB));
      myerr = 0.0;
   end

   function [myavg, myerr] = ubmeanABB(obsA, obsB)
      myavg = (1.0/(nsrcs*(nsrcs-1)*(nsrcs-2))) * (sum(obsA)*sum(obsB)^2 - sum(obsA)*sum(obsB.^2) - 2.0*sum(obsA.*obsB)*sum(obsB) + 2.0*sum(obsA.*(obsB.*obsB)));
      myerr = 0.0;
   end

   function [myavg, myerr] = ubmeanAAAA(obsA)
      e1 = sum(obsA);
      e2 = (1.0/2.0) * sum(obsA)^2 - (1.0/2.0) * sum(obsA.^2);
      e3 = (1.0/6.0) * sum(obsA)^3 - (1.0/6.0) * sum(obsA)*sum(obsA.^2) - (1.0/3.0) * sum(obsA)*sum(obsA.^2) + (1.0/3.0) * sum(obsA.^3);
      myavg = sum(obsA)*e3 - sum(obsA.^2)*e2 + sum(obsA.^3)*e1 - sum(obsA.^4);
      myavg = (6.0/(nsrcs*(nsrcs-1)*(nsrcs-2)*(nsrcs-3))) * myavg;
      myerr = 0.0;
   end

   foo1 = 0.5 * obs_l(:,2);
   foo2 = 0.25 * obs_s(:,2);
   foo3 = foo1 + foo2;
   foo4 = 0.5 * (obs_l(:,3) - obs_l(:,12));
   foo5 = 0.25 * (obs_s(:,3) - obs_s(:,12));
   foo6 = foo4 + foo5;
   foo7 = 0.5 * (obs_l(:,4) - 3.0*obs_l(:,13) + 2.0*obs_l(:,62));
   foo8 = 0.25 * (obs_s(:,4) - 3.0*obs_s(:,13) + 2.0*obs_s(:,62));
   foo9 = foo7 + foo8;
   foo10 = 0.5 * (obs_l(:,5) - 4.0*obs_l(:,14) - 3.0*obs_l(:,18) + 12.0*obs_l(:,63) - 6.0*obs_l(:,312));
   foo11 = 0.25 * (obs_s(:,5) - 4.0*obs_s(:,14) - 3.0*obs_s(:,18) + 12.0*obs_s(:,63) - 6.0*obs_s(:,312));
   foo12 = foo10 + foo11;
   foo13 = ms*0.25*obs_l(:,1);
   foo14 = ml*0.25*obs_s(:,1);
   foo15 = -0.25*ms*obs_l(:,11);
   foo16 = -0.25*ml*obs_s(:,11);

   nl = mean(foo1);
   delta_nl = ( std(real(foo1)) + 1.0j * std(imag(foo1)) ) / sqrt(nsrcs);
   ns = mean(foo2);
   delta_ns = ( std(real(foo2)) + 1.0j * std(imag(foo2)) ) / sqrt(nsrcs);
   n = mean(foo3);
   delta_n = ( std(real(foo3)) + 1.0j * std(imag(foo3)) ) / sqrt(nsrcs);

   nli = mean(foo4);
   delta_nli = ( std(real(foo4)) + 1.0j * std(imag(foo4)) ) / sqrt(nsrcs);
   nsi = mean(foo5);
   delta_nsi = ( std(real(foo5)) + 1.0j * std(imag(foo5)) ) / sqrt(nsrcs);
   ni = mean(foo6);
   delta_ni = ( std(real(foo6)) + 1.0j * std(imag(foo6)) ) / sqrt(nsrcs);

   nlii = mean(foo7);
   delta_nlii = ( std(real(foo7)) + 1.0j * std(imag(foo7)) ) / sqrt(nsrcs);
   nsii = mean(foo8);
   delta_nsii = ( std(real(foo8)) + 1.0j * std(imag(foo8)) ) / sqrt(nsrcs);
   nii = mean(foo9);
   delta_nii = ( std(real(foo9)) + 1.0j * std(imag(foo9)) ) / sqrt(nsrcs);

   nliii = mean(foo10);
   delta_nliii = ( std(real(foo7)) + 1.0j * std(imag(foo7)) ) / sqrt(nsrcs);
   nsiii = mean(foo11);
   delta_nsiii = ( std(real(foo8)) + 1.0j * std(imag(foo8)) ) / sqrt(nsrcs);
   niii = mean(foo12);
   delta_niii = ( std(real(foo9)) + 1.0j * std(imag(foo9)) ) / sqrt(nsrcs);

   [myavg, myerr]  = ubmeanAA(foo3);
   nn = myavg;
   delta_nn = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nn = n*n;
   end

   [myavg, myerr]  = ubmeanAB(foo3, foo6);
   nni = myavg;
   delta_nni = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nni = n*ni;
   end

   [myavg, myerr]  = ubmeanAAA(foo3);
   nnn = myavg;
   delta_nnn = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nnn = n*n*n;
   end

   [myavg, myerr]  = ubmeanAB(foo3, foo9);
   nnii = myavg;
   delta_nnii = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nnii = n*nii;
   end

   [myavg, myerr]  = ubmeanAA(foo6);
   nini = myavg;
   delta_nini = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nini = ni*ni;
   end

   [myavg, myerr]  = ubmeanABB(foo6, foo3);
   nnni = myavg;
   delta_nnni = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nnni = n*n*ni;
   end

   [myavg, myerr]  = ubmeanAAAA(foo3);
   nnnn = myavg;
   delta_nnnn = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nnnn = n*n*n*n;
   end

   [myavg, myerr]  = ubmeanAA(foo1);
   nlnl = myavg;
   delta_nlnl = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nlnl = nl*nl;
   end

   [myavg, myerr]  = ubmeanAA(foo2);
   nsns = myavg;
   delta_nsns = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nsns = ns*ns;
   end

   [myavg, myerr]  = ubmeanAB(foo1, foo2);
   nlns = myavg;
   delta_nlns = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nlns = nl*ns;
   end

   msccl = mean(foo13);
   delta_msccl = ( std(real(foo13)) + 1.0j * std(imag(foo13)) ) / sqrt(nsrcs);

   mlccs = mean(foo14);
   delta_mlccs = ( std(real(foo14)) + 1.0j * std(imag(foo14)) ) / sqrt(nsrcs);

   msccli = mean(foo15);
   delta_msccli = ( std(real(foo15)) + 1.0j * std(imag(foo15)) ) / sqrt(nsrcs);

   mlccsi = mean(foo16);
   delta_mlccsi = ( std(real(foo16)) + 1.0j * std(imag(foo16)) ) / sqrt(nsrcs);

   [myavg, myerr]  = ubmeanAB(foo3, foo13);
   nmsccl = myavg;
   delta_nmsccl = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nmsccl = n*msccl;
   end

   [myavg, myerr]  = ubmeanAB(foo3, foo14);
   nmlccs = myavg;
   delta_nmlccs = myerr;
   if (USE_BIASED_PRODUCT == 1)
      nmlccs = n*mlccs;
   end

if (TEST_ITERATIVE_FORMULAS == 1)
   % test AA
   display('BEGIN TEST')
   nn_ubnew = nn
   [myavg, myerr] = ubmean2(foo3, foo3);
   nn_ubold = myavg
   nn_naive = n*n

   % test AB
   display('')
   nni_ubnew = nni
   [myavg, myerr] = ubmean2(foo3, foo6);
   nni_ubold = myavg
   nni_naive = n*ni

   % test AAA
   display('')
   nn_ubnew = nnn
   [myavg, myerr] = ubmean3(foo3, foo3, foo3);
   nnn_ubold = myavg
   nnn_naive = n*n*n

   % test ABB
   display('')
   nnni_ubnew = nnni
   [myavg, myerr] = ubmean3(foo3, foo3, foo6);
   nnni_ubold = myavg
   nnni_naive = n*n*ni

   % test AAAA
   display('')
   nnnn_ubnew = nnnn
   [myavg, myerr] = ubmean4(foo3, foo3, foo3, foo3);
   nnnn_ubold = myavg
   nnnn_naive = n*n*n*n
   display('END TEST')
end

   % 01) nl
   % 02) ns
   % 03) n
   %
   % 04) nli
   % 05) nsi
   % 06) ni
   %
   % 07) nlii
   % 08) nsii
   % 09) nii
   %
   % 10) nliii
   % 11) nsiii
   % 12) niii
   %
   % 13) nn
   % 14) nni
   % 15) nnn
   %
   % 16) nnii
   % 17) nini
   % 18) nnni
   % 19) nnnn
   %
   % 20) nlnl
   % 21) nsns
   % 22) nlns
   %
   % 23) msccl
   % 24) mlccs
   %
   % 25) msccli
   % 26) mlccsi
   %
   % 27) nmsccl
   % 28) nmlccs

   res_avg = [nl, ns, n, nli, nsi, ni, nlii, nsii, nii, nliii, nsiii, niii, nn, nni, nnn, nnii, nini, nnni, nnnn, nlnl, nsns, nlns, msccl, mlccs, msccli, mlccsi, nmsccl, nmlccs];
   res_err = [delta_nl, delta_ns, delta_n, delta_nli, delta_nsi, delta_ni, delta_nlii, delta_nsii, delta_nii, delta_nliii, delta_nsiii, delta_niii, delta_nn, delta_nni, delta_nnn, delta_nnii, delta_nini, delta_nnni, delta_nnnn, delta_nlnl, delta_nsns, delta_nlns, delta_msccl, delta_mlccs, delta_msccli, delta_mlccsi, delta_nmsccl, delta_nmlccs];

end

