from sklearn.metrics import accuracy_score

def compute_metrics_gpt2(eval_preds, normalize=True, sample_weight=None):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels.flatten(), preds.flatten())
    return {'accuracy': acc}

# source : huggingface/run_clm.py
def compute_metrics_gpt2_hf(eval_preds, normalize=True, sample_weight=None):
    preds, labels = eval_preds

    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return {'accuracy': float(accuracy_score(labels, preds, normalize=normalize, sample_weight=sample_weight))}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors.
        # like past_key_values, but logits always come first
        logits= logits[0]
    return logits.argmax(dim=-1)
                                                
