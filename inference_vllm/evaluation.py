import os
import glob
import json
import jsonlines
import pandas as pd
import sklearn.metrics as sklm

def classification_report(pred,true, average="macro"):
    return {
        "accuracy" : sklm.accuracy_score(y_true=true, y_pred=pred),
        "f1-score" : sklm.f1_score(y_true=true, y_pred=pred, average=average),
        "precision" : sklm.precision_score(y_true=true, y_pred=pred, average=average, zero_division=0.0),
        "recall" : sklm.recall_score(y_true=true, y_pred=pred, average=average, zero_division=0.0)
    }

text2label = {"Yes":1,"No":0, -1:-1}

true_tup = [(el["id"],cid,el["answers"][cid]) for el in jsonlines.open('../data/test.jsonl') for cid in el["answers"]]
true_df = pd.DataFrame(true_tup, columns = ["file_id","consort_id","answer"]).sort_values(by=["file_id","consort_id"]).reset_index(drop=True)
true_df["label"] = true_df["answer"].apply(lambda x : text2label[x])

for pred_path in glob.glob("out/*/predictions.json"):
    predictions = json.load(open(pred_path))
    # skip if predictions are not finished
    if len(predictions) < len(true_tup) or os.path.exists(pred_path.replace("predictions.json","metrics.json")):
        continue
    
    predictions_tup = []
    for filename in predictions : 
        file_id, consort_id = filename.replace(".txt","").split("_")
        file_id = file_id.replace("@","/")
        predictions_tup.append((file_id,consort_id,predictions[filename]))
    pred_df = pd.DataFrame(predictions_tup, columns = ["file_id","consort_id","answer"]).sort_values(by=["file_id","consort_id"]).reset_index(drop=True)
    pred_df["label"] = pred_df["answer"].apply(lambda x : text2label[x])
    assert (true_df["file_id"] == pred_df["file_id"]).all()
    assert (true_df["consort_id"] == pred_df["consort_id"]).all()
    # Measure global micro performance
    micro_global = classification_report(pred_df["label"],true_df["label"])
    # Measure per-corpus performance
    covid_true = true_df[true_df["consort_id"].str.startswith('C', na=False)]["label"]
    depression_true = true_df[true_df["consort_id"].str.startswith('D', na=False)]["label"]
    covid_pred = pred_df[pred_df["consort_id"].str.startswith('C', na=False)]["label"]
    depression_pred = pred_df[pred_df["consort_id"].str.startswith('D', na=False)]["label"]
    covid_perf = classification_report(covid_pred,covid_true)
    depression_perf = classification_report(depression_pred,depression_true)
    # Measure global macro (corpus mean) performance
    macro_perf = {}
    for key in covid_perf :
        macro_perf[key] = (covid_perf[key] + depression_perf[key])/2
    # Measure per-criteria performance
    criteria_perfs = {}
    for crit in true_df["consort_id"].unique():
        crit_true = true_df[true_df["consort_id"] == crit]["label"]
        crit_pred = pred_df[pred_df["consort_id"] == crit]["label"]
        criteria_perfs[crit] = classification_report(crit_pred,crit_true)
    # Metrics dict
    metrics = {
        "micro_global":micro_global,
        "covid":covid_perf,
        "depression":depression_perf,
        "macro_global":macro_perf,
        "per_criteria":criteria_perfs
    }
    # Save metrics
    print("Saved ", pred_path)
    with open(pred_path.replace("predictions.json","metrics.json"),"w") as metric_outfile :
        json.dump(metrics,metric_outfile)