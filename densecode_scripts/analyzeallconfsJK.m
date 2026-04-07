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

function [res_avgs, res_errs] = analyzeallconfsJK(str, nruns, confs, srcs, JKBLOCKS, JKBLOCK, JKMODE, params)

   Ns = params(1);
   Nt = params(2);
   Nc = 3;
   T = params(3);

   res_avgs = [];
   res_errs = [];

   if (JKMODE == 0)
      JKINT = confs;
      JKLEN = floor( ( length(JKINT) / JKBLOCKS ) )
      JKINT = [ JKINT(1:(JKBLOCK-1)*JKLEN), JKINT(1+JKBLOCK*JKLEN:JKBLOCKS*JKLEN) ]

      for i = JKINT
         fname = strcat(str,num2str(i));
         [r1, r2] = analyze1conf(fname, nruns, srcs, params);
         res_avgs = [res_avgs; r1];
         res_errs = [res_errs; r2];
      end
   elseif (JKMODE == 1)
      display('FOR TESTING PURPOSES')
   end

   %%%

   display('----------')

   % https://arxiv.org/pdf/2001.08530.pdf eq. 2
   % https://arxiv.org/pdf/1701.04325.pdf eq. 9

   dlnZdmu = mean( res_avgs(:,3) );

   d2lnZdmu2 =  - mean( res_avgs(:,3) )^2 + mean( res_avgs(:,6) ) + mean( res_avgs(:,13) );

   chi3B_part1 = - 3.0 * dlnZdmu * d2lnZdmu2;
   chi3B_part2 = - mean( res_avgs(:,3) )^3;
   chi3B_part3 = mean( res_avgs(:,15) ) + 3.0 * mean( res_avgs(:,14) ) + mean( res_avgs(:,9) );

   d3lnZdmu3 = (chi3B_part1 + chi3B_part2 + chi3B_part3);

   chi4B_part1 = - 3.0 * d2lnZdmu2 * d2lnZdmu2 - 3.0 * dlnZdmu * d3lnZdmu3 - 3.0 * dlnZdmu * dlnZdmu * d2lnZdmu2;
   chi4B_part2 = - mean( res_avgs(:,3) ) * ( mean( res_avgs(:,15) ) + 3.0 * mean( res_avgs(:,14) ) + mean( res_avgs(:,9) ) );
   chi4B_part3 = 6.0 * mean( res_avgs(:,18) ) + 4.0 * mean( res_avgs(:,16) ) + 3.0 * mean( res_avgs(:,17) ) + mean( res_avgs(:,12) ) + mean( res_avgs(:,19) );

   d4lnZdmu4 = (chi4B_part1 + chi4B_part2 + chi4B_part3);

   chi1B = (Nt^3 / Ns^3) * (1.0/(3.0*Nt)) * dlnZdmu
   chi2B = (Nt^3 / Ns^3) * (1.0/(3.0*Nt)^2) * d2lnZdmu2
   chi3B = (Nt^3 / Ns^3) * (1.0/(3.0*Nt)^3) * d3lnZdmu3
   chi4B = (Nt^3 / Ns^3) * (1.0/(3.0*Nt)^4) * d4lnZdmu4

   chi1Q = (Nt^3 / Ns^3) * (1.0/(3.0*Nt)) * (0.5*mean( res_avgs(:,1) ) - mean( res_avgs(:,2) ))

   chi1Qi = (Nt^3 / Ns^3) * (1.0/(9.0*Nt*Nt));

   chi1Qi_part1 = - 0.5*mean( res_avgs(:,1) )^2 + 0.5*mean( res_avgs(:,20) ) + 0.5*mean( res_avgs(:,4) );
   chi1Qi_part2 =       mean( res_avgs(:,2) )^2 -     mean( res_avgs(:,21) ) -     mean( res_avgs(:,5) );
   chi1Qi_part3 =   0.5*mean( res_avgs(:,1) )*mean( res_avgs(:,2) ) - 0.5 * mean( res_avgs(:,22) );

   chi1Qi = chi1Qi * (   chi1Qi_part1 + chi1Qi_part2 + chi1Qi_part3   )

   ffpi4 = ((156.1/sqrt(2))/(Nt*T))^4;

   cc = mean(res_avgs(:,23) ) - mean(res_avgs(:,24) );
   cc = (1.0 / (Nt * Ns^3)) * 2.0*cc/ffpi4

   cci = - mean(res_avgs(:,3) ) * mean(res_avgs(:,23) ) + mean(res_avgs(:,25) ) + mean(res_avgs(:,27) );
   cci = cci - (   - mean(res_avgs(:,3) ) * mean(res_avgs(:,24) ) + mean(res_avgs(:,26) ) + mean(res_avgs(:,28) )   );
   cci = (1.0 / (Nt * Ns^3)) * 2.0*cci/ffpi4;
   cci = (1.0/(3.0*Nt)) * cci

   display('----------')

   %%%

   display('----------')

   chi001uds = mean(res_avgs(:,2));
   chi001uds = (Nt^3 / Ns^3) * (1.0/(Nt)) * chi001uds

   chi010uds = mean(0.5*res_avgs(:,1));
   chi010uds = (Nt^3 / Ns^3) * (1.0/(Nt)) * chi010uds

   chi100uds = chi010uds

   chi1B_BLFLD = (1.0/3.0) * ( chi001uds + chi010uds + chi010uds )

   chi002uds = - mean(res_avgs(:,2))^2  + mean(res_avgs(:,5)) + mean(res_avgs(:,21));
   chi002uds = chi002uds * (Nt^3 / Ns^3) * (1.0/(Nt)^2)

   chi020uds = - mean(0.5*res_avgs(:,1))^2  + mean(0.5*res_avgs(:,4)) + mean(0.25*res_avgs(:,20));
   chi020uds = chi020uds * (Nt^3 / Ns^3) * (1.0/(Nt)^2)

   chi200uds = chi020uds

   chi011uds = - mean(0.5*res_avgs(:,1)) * mean(res_avgs(:,2)) + mean(0.5*res_avgs(:,22));
   chi011uds = chi011uds * (Nt^3 / Ns^3) * (1.0/(Nt)^2)

   chi101uds = chi011uds

   chi110uds = - mean(0.5*res_avgs(:,1))^2 + mean(0.25*res_avgs(:,20));
   chi110uds = chi110uds * (Nt^3 / Ns^3) * (1.0/(Nt)^2)

   chi2B_BLFLD = (1.0/9.0) * ( chi002uds + chi020uds + chi200uds + 2*(chi011uds + chi101uds + chi110uds) )

   display('----------')

   %%%

end
