for i in JK* ; do
    echo $i
    cd $i
    cat *out | grep 'chi1B = ' | awk -e '{print $3" " $4$5}' > dati0_chi1B.txt
    cat *out | grep 'chi2B = ' | awk -e '{print $3" " $4$5}' > dati0_chi2B.txt
    cat *out | grep 'chi3B = ' | awk -e '{print $3" " $4$5}' > dati0_chi3B.txt
    cat *out | grep 'chi4B = ' | awk -e '{print $3" " $4$5}' > dati0_chi4B.txt
    cat *out | grep 'chi1Q = ' | awk -e '{print $3" " $4$5}' > dati0_chi1Q.txt
    cat *out | grep 'chi1Qi = ' | awk -e '{print $3" " $4$5}' > dati0_chi1Qi.txt
    cat *out | grep 'cc = ' | awk -e '{print $3" " $4$5}' > dati0_cc.txt
    cat *out | grep 'cci = ' | awk -e '{print $3" " $4$5}' > dati0_cci.txt
    cd ..
done



for i in JK*_s{?,??}/*dati0_chi1B.txt ; do
   echo $i
   octave dojk.m $i
done



rm foo



cat JK*_s{?,??}/*res.txt > foo
cp header.txt res.txt
paste -d " " mu.txt foo >> res.txt



rm foo

