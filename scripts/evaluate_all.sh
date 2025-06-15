for f in *c2j*.json; do
  python3 code/utils/evaluator.py --mode c2j data/binary/test.csv "$f" --out results.csv
done

for f in *j2c*.json; do
  python3 code/utils/evaluator.py --mode j2c data/binary/test.csv "$f" --out results.csv
done
