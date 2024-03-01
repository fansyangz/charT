from transformers import TrainingArguments


class Arguments:
    def __init__(self):
        raise EnvironmentError("can not to be instantiated")

    @classmethod
    def arguments(cls, output_dir, num_train_epochs, batch_size, seed):
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            remove_unused_columns=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            do_train=True,
            do_eval=True,
            do_predict=True,
            seed=seed,
            report_to=["none"],
            save_strategy="no"
        )
