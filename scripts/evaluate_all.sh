DATA_DIR=$1
RES_DIR=$2

for f in $RES_DIR/*c2j*.pkl.gz; do
  python3 code/utils/evaluator.py --mode c2j $DATA_DIR/binary/test.csv "$f" --out $RES_DIR/results.csv --candidates $DATA_DIR/candidates.csv --jobs $DATA_DIR/jobs.csv --train $DATA_DIR/binary/train.csv
done

for f in $RES_DIR/*j2c*.pkl.gz; do
  python3 code/utils/evaluator.py --mode j2c $DATA_DIR/binary/test.csv "$f" --out $RES_DIR/results.csv --candidates $DATA_DIR/candidates.csv --jobs $DATA_DIR/jobs.csv --train $DATA_DIR/binary/train.csv

done
