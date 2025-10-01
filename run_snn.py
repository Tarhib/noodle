import os
import time
import utils.metrics_snn as metrics
import torch
import faiss
import numpy as np

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_knn_func(in_dataset, model_arch, out_datasets, start, end,M_actual):
        in_dataset = in_dataset
        model_arch = model_arch
        cache_name = f"cache/{in_dataset}_{model_arch}_train_in_alllayers.npy"
        feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
        feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
        class_num = score_log.shape[1]
        start = start; stop = end;
        cache_name = f"cache/{in_dataset}_{model_arch}_val_in_alllayers.npy"
        feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
        feat_log_val, score_log_val = feat_log_val.T.astype(np.float32), score_log_val.T.astype(np.float32)

        ood_feat_log_all = {}
        for ood_dataset in out_datasets:
            cache_name = f"cache/{ood_dataset}vs{in_dataset}_{model_arch}_out_alllayers.npy"
            ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
            ood_feat_log, ood_score_log = ood_feat_log.T.astype(np.float32), ood_score_log.T.astype(np.float32)
            ood_feat_log_all[ood_dataset] = ood_feat_log

        #normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

        #prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(start, stop)]))
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

        ftrain = prepos_feat(feat_log)
        ftest = prepos_feat(feat_log_val)
        food_all = {}
        for ood_dataset in out_datasets:
            food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

        index = faiss.IndexFlatL2(ftrain.shape[1])
        index.add(ftrain)
        for K in [20]:

            D, _ = index.search(ftest, K)
            scores_in = -D[:,-1]
            
            print(scores_in)
            output_file_name = "scores_in.txt"
        
            # Write the scores to the file
            with open(output_file_name, "w") as file:
                for score in scores_in:
                    file.write(f"{score}\n")
            all_results = []
            all_score_ood = []
            for ood_dataset, food in food_all.items():
                D, _ = index.search(food, K)
                scores_ood_test = -D[:,-1]
                print("Score OOD")
                print(scores_ood_test)
                output_file_name = "scores_ood_"+ood_dataset+".txt"
        
                # Write the scores to the file
                with open(output_file_name, "w") as file:
                  for score in scores_ood_test:
                      file.write(f"{score}\n")
                all_score_ood.extend(scores_ood_test)
                results = metrics.cal_metric(scores_in, scores_ood_test)
                all_results.append(results)

            metrics.print_all_results(all_results, out_datasets, f'Pseudo Label Induced Subspace with M={M_actual} labels')
            print()
