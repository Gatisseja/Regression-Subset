#!/bin/bash
#In this directory, 1.csv, 2.csv files

if [ ! -d Figures ]; then
  mkdir Figures
fi


for i in `seq 1 10`; do


       echo "Creating File 'Figures/${i}_Sharp_v_AnnRet.png'"
gnuplot << EOF
          reset
          set key off
	  set datafile separator ","

	  set terminal pngcairo enhanced font 'Verdana,15'


          set title 'Sharpe Ratio vs Annual Return for ${i} predictors'
          set xlabel 'Annual Return (%)'
          set ylabel 'Sharpe Ratio'


          set output 'Figures/${i}_Sharp_v_AnnRet.png'

          plot '${i}.csv' using 1:3 every 3
EOF

       echo "Creating File 'Figures/${i}_Rsq_v_AnnRet.png'"
gnuplot << EOF
          reset
          set key off
	  set datafile separator ","

	  set terminal pngcairo enhanced font 'Verdana,15'


          set title 'Adj R-squared vs Annual Return for ${i} predictors'
          set xlabel 'Annual Return (%)'
          set ylabel 'Adj R-squared'


          set output 'Figures/${i}_Rsq_v_AnnRet.png'



          plot '${i}.csv' using 1:4 every 3
EOF
done

exit 0
