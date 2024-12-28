from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field

@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    batch_eval_metrics: bool = field(default=False, metadata={"help": "Whether to evaluate metrics per batch."})
    do_train: bool = field(default=False, metadata={"help": "Whether to evaluate metrics per batch."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to evaluate metrics per batch."})
    output_dir: str = field(
        default="/Users/a58/Downloads/my_test/my_code/output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs to perform."},
    )