from pipeline.pipeline_runner import PipelineRunner
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--out", default="results")
parser.add_argument("--model", default="rf")
parser.add_argument("--cv", default="lopo")

args = parser.parse_args()

runner = PipelineRunner(
    data_dir=args.data_dir,
    model_name=args.model,
    cv_method=args.cv,
    output_dir=args.out
)

preds, labels = runner.run_node_level_predictor()
print("Done.")
