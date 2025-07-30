import argparse
import ast
import os.path

import pandas as pd
import numpy as np
from tqdm import tqdm

from cv_parser import parse_resume
from description_parser import parse_description


def parse_args():
    parser = argparse.ArgumentParser(description="Paramètres de parallélisation.")

    # general params
    parser.add_argument('--num_split', type=int, default=1,
                        help="Nombre de split voulus pour le dataset. Ex: 4 prend en compte 1/4 du dataset.")
    parser.add_argument('--idx_split', type=int, default=1,
                        help="Indice du split voulu. Ex: 2 prend le 2ème 1/4 du dataset.")
    parser.add_argument("--type", type=str, default="cv",
                        help="cv or job")
    parser.add_argument("--id", type=str, default=None,
                        help="If given, id of the candidate to consider.")

    args = parser.parse_args()
    return args


def process_df_job(df, id):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        job_id = row['job_id']
        if id is not None and str(job_id) != str(id):
            continue
        if not os.path.isfile(f"annonce_infos/annonce_{job_id}.json") or id is not None:
            try:
                descr = str(row['description'])
                resp = parse_description(descr)

                with open(f"annonce_infos/annonce_{job_id}.json", "w") as out:
                    out.write(resp)
            except Exception as e:
                print(e)
                print(f'Not able to process job offer {job_id}, adding it to retry pile.')


def find_end(s):
    prev = ""
    for i, x in enumerate(s):
        if x == "'" and prev != "\\":
            return i
        prev = x
    return -1


def read_string(s):
    s = str(s).strip()
    if s.startswith("b\'"):
        end = find_end(s[2:]) + 3
        s = ast.literal_eval(s[:end]).replace(b"\r", b"\n").decode("latin-1") + "\n" + s[end:]
    s = s.strip().rstrip("Volontariat").replace("\\n", "\n").replace("\\\"", "\"")
    if s.startswith("Unable to extract PDF"):
        return ""
    else:
        return s


def process_df_cv(df, id):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cand_id = row['candidate_id']
        if id is not None and str(cand_id) != str(id):
            continue
        if not os.path.isfile(f"all_cvs/cv_{cand_id}.json") or id is not None:
            try:
                resume = read_string(row['resume_info']).strip() if not pd.isna(row['resume_info']) else ""
                short = len(resume) < 100
                lastname, firstname = str(row['lastname']), str(row['firstname'])
                resume = f'Name:{firstname}, {lastname}\n{resume}'

                resp = parse_resume(resume, short=short)

                with open(f"all_cvs/cv_{cand_id}.json", "w") as out:
                    out.write(resp)

            except Exception as e:
                print(e)
                print(f'Not able to process resume {cand_id}, adding it to retry pile.')


def main():
    args = parse_args()
    nb_split = args.num_split
    id_split = args.idx_split
    id_candidate = args.id

    if nb_split < 1:
        raise ValueError('Le dataset doit être divisé en un nombre positif de splits.')
    elif nb_split < id_split or id_split <= 0:
        raise ValueError('Indice du split out of range (les indices commencent à 1).')
    else:
        print(f"Processing split n°{id_split} over {nb_split} splits...")
        if args.type == 'cv':
            df = pd.read_csv('data/candidates_final.csv')
            length = len(df)

            c = ([int(np.floor(length / nb_split)) for _ in range(nb_split - 1)] +
                 [int(np.floor(length / nb_split) + length % nb_split)])
            c = [sum(c[:i]) for i in range(len(c) + 1)]
            subdf = [df.iloc[c[i]:c[i + 1]] for i in range(len(c) - 1)]

            df_inproc = subdf[id_split - 1]

            process_df_cv(df_inproc, id_candidate)
        else:
            df = pd.read_csv('data/jobs_final.csv')
            length = len(df)
            c = [int(np.floor(length / nb_split)) for _ in range(nb_split - 1)] + [
                int(np.floor(length / nb_split) + length % nb_split)]
            c = [sum(c[:i]) for i in range(len(c) + 1)]
            subdf = [df.iloc[c[i]:c[i + 1]] for i in range(len(c) - 1)]

            df_inproc = subdf[id_split - 1]

            process_df_job(df_inproc, id_candidate)


if __name__ == "__main__":
    main()
