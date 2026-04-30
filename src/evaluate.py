from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
import wandb
import yaml

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

def run_evaluation(pipeline, eval_questions: list[dict]):
    config = load_config()

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        job_type="evaluation"
    )

    results = []
    for item in eval_questions:
        output = pipeline.query(item["question"])
        results.append({
            "question": item["question"],
            "answer": output["answer"],
            "contexts": output["contexts"],
            "ground_truth": item["ground_truth"]
        })

    dataset = Dataset.from_list(results)
    scores = evaluate(dataset, metrics=[
        faithfulness,
        answer_relevancy,
        context_recall
    ])

    # W&B: log all RAGAS scores
    wandb.log({
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "context_recall": scores["context_recall"],
    })

    # W&B: log results table
    table = wandb.Table(
        columns=["question", "answer", "faithfulness",
                 "answer_relevancy", "context_recall"]
    )
    wandb.log({"eval_results": table})
    wandb.finish()
    return scores