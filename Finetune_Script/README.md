# Fine-tuning LLMs on three unit testing tasks

We use a general training script for the same architecture. The name of the training script consists of two parts: the first part is the architecture name, including `encoder`, `encoder-decoder`, and `decoder`. The second part is the task, including `ag` (assertion generation), `tg` (test generation), and `tu` (test evolution).

Additionally, we have created different training startup scripts for different models to facilitate model training, provided that the training set is properly constructed.
