DATA_DIR=$1
RES_DIR=$2

python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/forest_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model forest --use_temporal_features --use_llm_features
python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/forest_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model forest --use_temporal_features --use_llm_features

python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model logreg --use_temporal_features --use_llm_features
python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model logreg --use_temporal_features --use_llm_features

python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_baseline_nr_10_j2c.json --mode j2c --neg_ratio 10.0 --model logreg --use_temporal_features --use_llm_features
python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_baseline_nr_10_c2j.json --mode c2j --neg_ratio 10.0 --model logreg --use_temporal_features --use_llm_features

python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/forest_no_temporal_features_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model forest --use_llm_features
python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/forest_no_temporal_features_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model forest --use_llm_features

python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_temporal_features_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model logreg --use_llm_features
python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_temporal_features_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model logreg --use_llm_features

python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_llm_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model logreg --use_temporal_features
python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_llm_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model logreg --use_temporal_features

python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_llm_no_temporal_features_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model logreg
python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_llm_no_temporal_features_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model logreg

python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_temporal_features_no_human_features_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model logreg --use_llm_features --remove_human_features
python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_temporal_features_no_human_features_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model logreg --use_llm_features --remove_human_features


python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_human_features_baseline_j2c.json --mode j2c --neg_ratio 1.0 --model logreg --use_temporal_features --use_llm_features --remove_human_features
python code/baselines/feature_classifier_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/logreg_no_human_features_baseline_c2j.json --mode c2j --neg_ratio 1.0 --model logreg --use_temporal_features --use_llm_features --remove_human_features

python code/baselines/random_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/random_baseline_j2c.json --mode j2c
python code/baselines/random_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/random_baseline_c2j.json --mode c2j

python code/baselines/temporal_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/temporal_baseline_j2c.json --mode j2c
python code/baselines/temporal_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/temporal_baseline_c2j.json --mode c2j

python code/baselines/past_temporal_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/past_temporal_baseline_c2j.json --mode c2j
python code/baselines/past_temporal_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/past_temporal_baseline_j2c.json --mode j2c

python code/baselines/collab_cf_temporal.py $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/cf_baseline_c2j.json --mode c2j
python code/baselines/collab_cf_temporal.py $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/cf_baseline_j2c.json --mode j2c

python code/baselines/collab_mf_temporal.py $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/mf_baseline_c2j.json --mode c2j --dim 64 --init_lr 0.05 --reg 0.02 --epochs 4
python code/baselines/collab_mf_temporal.py $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/mf_baseline_j2c.json --mode j2c --dim 64 --init_lr 0.05 --reg 0.02 --epochs 4

python code/baselines/online_popularity_baseline.py $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/pop_online_baseline_c2j.json --mode c2j
python code/baselines/online_popularity_baseline.py $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/pop_online_baseline_j2c.json --mode j2c

python code/baselines/popularity_recency_baseline.py $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $RES_DIR/poprec_preds_baseline_c2j.json \
       --mode c2j --lambda 0.015
python code/baselines/popularity_recency_baseline.py $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $RES_DIR/poprec_preds_baseline_j2c.json \
       --mode j2c --lambda 0.015

python code/baselines/jaccard_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/jaccard_baseline_c2j.json --mode c2j --use_human_features --use_llm_features
python code/baselines/jaccard_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/jaccard_baseline_j2c.json --mode j2c --use_human_features --use_llm_features

python code/baselines/jaccard_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/jaccard_baseline_no_human_c2j.json --mode c2j --use_llm_features
python code/baselines/jaccard_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/jaccard_baseline_no_human_j2c.json --mode j2c --use_llm_features

python code/baselines/jaccard_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/jaccard_baseline_no_llm_c2j.json --mode c2j --use_human_features
python code/baselines/jaccard_baseline.py $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $DATA_DIR/binary/test.csv $RES_DIR/jaccard_baseline_no_llm_j2c.json --mode j2c --use_human_features

python code/baselines/hybrid_recency_cf_baseline.py \
       $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $RES_DIR/recCF_baseline_c2j.json \
       --mode c2j --max_age 90
python code/baselines/hybrid_recency_cf_baseline.py \
       $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $DATA_DIR/candidates.csv $DATA_DIR/jobs.csv $RES_DIR/recCF_baseline_j2c.json \
       --mode j2c --max_age 90

python code/baselines/fm_online_temporal_baseline.py \
       $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/fm_baseline_c2j.json \
       --mode c2j --neg_ratio 1 --dim 64 --lr 0.03 --reg 0.02 --epochs 3
python code/baselines/fm_online_temporal_baseline.py \
       $DATA_DIR/binary/train.csv $DATA_DIR/binary/test.csv $RES_DIR/fm_baseline_j2c.json \
       --mode j2c --neg_ratio 1 --dim 64 --lr 0.03 --reg 0.02 --epochs 3
