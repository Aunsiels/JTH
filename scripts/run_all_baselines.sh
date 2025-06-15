python code/baselines/random_baseline.py data/candidates.csv data/jobs.csv data/binary/test.csv random_baseline_j2c.json --mode j2c
python code/baselines/random_baseline.py data/candidates.csv data/jobs.csv data/binary/test.csv random_baseline_c2j.json --mode c2j

python code/baselines/temporal_baseline.py data/candidates.csv data/jobs.csv data/binary/test.csv temporal_baseline_j2c.json --mode j2c
python code/baselines/temporal_baseline.py data/candidates.csv data/jobs.csv data/binary/test.csv temporal_baseline_c2j.json --mode c2j

python code/baselines/past_temporal_baseline.py data/candidates.csv data/jobs.csv data/binary/test.csv past_temporal_baseline_c2j.json --mode c2j
python code/baselines/past_temporal_baseline.py data/candidates.csv data/jobs.csv data/binary/test.csv past_temporal_baseline_j2c.json --mode j2c

python code/baselines/collab_cf_temporal.py data/binary/train.csv data/binary/test.csv cf_baseline_c2j.json --mode c2j
python code/baselines/collab_cf_temporal.py data/binary/train.csv data/binary/test.csv cf_baseline_j2c.json --mode j2c

python code/baselines/collab_mf_temporal.py data/binary/train.csv data/binary/test.csv mf_baseline_c2j.json --mode c2j --dim 64 --init_lr 0.05 --reg 0.02 --epochs 4
python code/baselines/collab_mf_temporal.py data/binary/train.csv data/binary/test.csv mf_baseline_j2c.json --mode j2c --dim 64 --init_lr 0.05 --reg 0.02 --epochs 4

python code/baselines/feature_classifier_baseline.py data/candidates.csv data/jobs.csv data/binary/train.csv data/binary/test.csv forest_no_temporal_features_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model forest
python code/baselines/feature_classifier_baseline.py data/candidates.csv data/jobs.csv data/binary/train.csv data/binary/test.csv forest_no_temporal_features_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model forest

python code/baselines/feature_classifier_baseline.py data/candidates.csv data/jobs.csv data/binary/train.csv data/binary/test.csv logreg_no_temporal_features_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model logreg
python code/baselines/feature_classifier_baseline.py data/candidates.csv data/jobs.csv data/binary/train.csv data/binary/test.csv logreg_no_temporal_features_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model logreg

python code/baselines/feature_classifier_baseline.py data/candidates.csv data/jobs.csv data/binary/train.csv data/binary/test.csv forest_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model forest --use_temporal_features
python code/baselines/feature_classifier_baseline.py data/candidates.csv data/jobs.csv data/binary/train.csv data/binary/test.csv forest_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model forest --use_temporal_features

python code/baselines/feature_classifier_baseline.py data/candidates.csv data/jobs.csv data/binary/train.csv data/binary/test.csv logreg_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model logreg --use_temporal_features
python code/baselines/feature_classifier_baseline.py data/candidates.csv data/jobs.csv data/binary/train.csv data/binary/test.csv logreg_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model logreg --use_temporal_features

python code/baselines/online_popularity_baseline.py data/binary/train.csv data/binary/test.csv pop_online_baseline_c2j.json --mode c2j
python code/baselines/online_popularity_baseline.py data/binary/train.csv data/binary/test.csv pop_online_baseline_j2c.json --mode j2c

python code/baselines/popularity_recency_baseline.py data/binary/train.csv data/binary/test.csv data/candidates.csv data/jobs.csv poprec_preds_baseline_c2j.json \
       --mode c2j --lambda 0.015
python code/baselines/popularity_recency_baseline.py data/binary/train.csv data/binary/test.csv data/candidates.csv data/jobs.csv poprec_preds_baseline_j2c.json \
       --mode j2c --lambda 0.015

python code/baselines/jaccard_baseline.py data/candidates.csv data/jobs.csv data/binary/test.csv jaccard_baseline_c2j.json --mode c2j
python code/baselines/jaccard_baseline.py data/candidates.csv data/jobs.csv data/binary/test.csv jaccard_baseline_j2c.json --mode j2c

python code/baselines/hybrid_recency_cf_baseline.py \
       data/binary/train.csv data/binary/test.csv data/candidates.csv data/jobs.csv recCF_baseline_c2j.json \
       --mode c2j --max_age 90
python code/baselines/hybrid_recency_cf_baseline.py \
       data/binary/train.csv data/binary/test.csv data/candidates.csv data/jobs.csv recCF_baseline_j2c.json \
       --mode j2c --max_age 90

python code/baselines/fm_online_temporal_baseline.py \
       data/binary/train.csv data/binary/test.csv fm_baseline_c2j.json \
       --mode c2j --neg_ratio 1 --dim 64 --lr 0.03 --reg 0.02 --epochs 3
python code/baselines/fm_online_temporal_baseline.py \
       data/binary/train.csv data/binary/test.csv fm_baseline_j2c.json \
       --mode j2c --neg_ratio 1 --dim 64 --lr 0.03 --reg 0.02 --epochs 3
