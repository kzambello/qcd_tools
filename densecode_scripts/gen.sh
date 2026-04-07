DIRNAME="beta_6p120_Nt_6"
MAXCONF="50000"
MINCONF="1000"
NSRCS="500"
NRUNS="1"
NS="36"
NT="6"
TEMP="136.1"
NMUS="19"

for ((iMU=1; iMU <= $NMUS; iMU = iMU+1)); do
   mkdir  JK20_s"$iMU"
   for iBLOCK in {1..20}; do
      echo "iMU = $iMU , iBLOCK = $iBLOCK"
      cat fai.m.template | sed -e "s/SED_DIRNAME/$DIRNAME/g" | sed -e "s/SED_MU/$iMU/g" | sed -e "s/SED_MAXCONF/$MAXCONF/g" | sed -e "s/SED_MINCONF/$MINCONF/g" | sed -e "s/SED_NSRCS/$NSRCS/g" | sed -e "s/SED_NRUNS/$NRUNS/g" | sed -e "s/SED_BLOCK/$iBLOCK/g" | sed -e "s/SED_NS/$NS/g" | sed -e "s/SED_NT/$NT/g" | sed -e "s/SED_TEMP/$TEMP/g" > JK20_s"$iMU"/fai_"$(($iBLOCK-1))".m
      cp an*m JK20_s"$iMU"/
   done
done


cat go.template | sed -e "s/SED_NMUS/$NMUS/g" > go
cat start.sh.template | sed -e "s/SED_DIRNAME/$DIRNAME/g" > start.sh

echo "0.00000
0.02182
0.04363
0.06545
0.08727
0.10908
0.13090
0.15272
0.16362
0.17453
0.010910
0.032725
0.054540
0.076360
0.098175
0.119990
0.141810
0.158170
0.169075" > mu.txt
