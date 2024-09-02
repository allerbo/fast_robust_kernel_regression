for nu in 100 0.5 1.5 2.5 10
do
  for cauchy in 0 0.1
  do
    for data in house super temp steel airfoil
    do
      sbatch --array=1-50 start_real.sh cauchy=$cauchy data=\"$data\" nu=$nu
    done
  done
done
