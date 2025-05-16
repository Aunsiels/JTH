# JTH

This repository contains the code for the paper _JTH: A Dataset for Evaluating Cold-Start and Temporal Dynamics
in Job Recommendation_. Please cite our work with:

```
@article{millet2025,
  title={JTH: A Dataset for Evaluating Cold-Start and Temporal Dynamics
in Job Recommendation},
  author={Millet, Yann and Behar, Éric and Romero, Julien}
  year={2025}
}
```

## Structure

You can find the code of the baselines in _code/baselines_. The binary splitter
and the evaluation script is in _code/utils_.

## Output Structure

To use the evaluation script, the input predictions must be a JSON. This JSON
is a dictionary where the key is a pair of (candidate id, timestamp), and the value
is the ranked list of the recommended job ids.