import torch
import numpy as np
import utils.metrics_snn as metrics
from sklearn.metrics.pairwise import cosine_similarity
import models.densenet as dn
import torch.nn.functional as F
import models.ood_detect as ood_detect

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_ood_detection_with_confusion_projection(in_dataset, model_arch, out_datasets, start, end):
    # Load ID distances (e.g., from val split, since that’s used for OOD evaluation)
    dist_cache_name_in = f"cache/{in_dataset}_{model_arch}_val_in_distances.npy"
    scores_in = np.load(dist_cache_name_in)
    print(scores_in)
    output_file_name = "scores_in.txt"

    # Write the scores to the file
    with open(output_file_name, "w") as file:
        for score in scores_in:
            file.write(f"{score}\n")

    all_results = []
    for ood_dataset in out_datasets:
        dist_cache_name_ood = f"cache/{ood_dataset}vs{in_dataset}_{model_arch}_out_distances.npy"
        scores_ood = np.load(dist_cache_name_ood)
        print("Score OOD")
        output_file_name = "scores_ood_"+ood_dataset+".txt"

        # Write the scores to the file
        with open(output_file_name, "w") as file:
            for score in scores_ood:
                file.write(f"{score}\n")


        print(scores_ood)
        # Now we have scores_in (ID) and scores_ood (OOD)
        # Use metrics from earlier
        #results = metrics.cal_metric(scores_in, scores_ood)
        #all_results.append(results)
        results = metrics.cal_metric(scores_in, scores_ood)
        all_results.append(results)

    metrics.print_all_results(all_results, out_datasets, f'SNN k=20')


from sklearn.metrics import roc_curve, auc
import numpy as np

def cal_metric(known, novel):
    results = dict()

    # Concatenate known and novel scores
    y_true = np.concatenate([np.ones_like(known), np.zeros_like(novel)])
    y_scores = np.concatenate([known, novel])

    # Use sklearn to calculate FPR, TPR, thresholds for AUROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate AUROC using sklearn's AUC function
    results['AUROC'] = auc(fpr, tpr)

    # FPR at TPR = 95%
    target_tpr = 0.95
    idx = np.argmin(np.abs(tpr - target_tpr))
    results['FPR'] = fpr[idx] if idx < len(fpr) else 1.0

    # Detection error at 95% TPR
    results['DTERR'] = ((1 - tpr[idx]) * 0.5 + fpr[idx] * 0.5) if idx < len(fpr) else 1.0

    # Area Under Inverse Precision-Recall curve (AUIN)
    precision = tpr / (tpr + fpr + 1e-10)  # To avoid division by zero
    results['AUIN'] = auc(tpr, precision)

    # Area Under the OUT distribution curve (AUOUT)
    out_precision = (1 - fpr) / ((1 - fpr) + (1 - tpr) + 1e-10)
    results['AUOUT'] = auc(1 - fpr, out_precision)

    return results

# Function to compute average results
def compute_average_results(all_results):
    mtypes = ['FPR', 'AUROC', 'AUIN']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results

# Function to print results in the desired format
def print_all_results(results, datasets, method):
    mtypes = ['FPR', 'AUROC', 'AUIN']
    avg_results = compute_average_results(results)
    
    # Header
    print(f' OOD detection method: {method}')
    print(f'{"Dataset":<12} {"FPR":>6} {"AUROC":>6} {"AUIN":>6}')

    # Print results for each OOD dataset
    for result, dataset in zip(results, datasets):
        print(f'{dataset:<12} {result["FPR"] * 100:6.2f} {result["AUROC"] * 100:6.2f} {result["AUIN"] * 100:6.2f}')

    # Print average results
    #print(f'{"AVG":<12} {avg_results["FPR"] * 100:6.2f} {avg_results["AUROC"] * 100:6.2f} {avg_results["AUIN"] * 100:6.2f}')
    print()

