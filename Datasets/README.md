# Datasets

This fold append datasets for three unit testing tasks: test generation, assertion generation, and test evolution.

## Test Generation

In the test generation task, we utilize {Methods2Test} dataset proposed by Tufano et al. to fine-tune LLMs. This dataset is created through an extensive mining process, resulting in 780,944 pairs of JUnit tests and focal methods. However, due to resource constraints, we are unable to use the entire dataset. To reduce the dataset and enhance its quality, we filter the dataset by several rules and obtain 56,132 pairs of data and call this dataset $Method2Test_{filter}$. We then split the filtered dataset by repositories into training, validation, and test sets using an 8:1:1 ratio for further fine-tuning.

## Assertion Generation

In the assertion generation task, we use the dataset known as $Data_{old}$. This dataset is derived from a raw dataset used by *ATLAS*. Each entry in $Data_{old}$ is referred to as a Test-Assert Pair (TAP). A TAP consists of two components: (1) a focal-test pair, which includes a test method without an assertion and its corresponding focal method; (2) assertions. 
To simplify the problem, $Data_{old}$ excludes any TAP where the assertions contain tokens that are not present in the focal-test pair. In total, $Data_{old}$ contains 156,760 TAPs, which are divided into training, validation, and test sets by the ratio of 8:1:1 for further fine-tuning.

## Test Evolution

In the test evolution task, we utilize the dataset proposed by Hu et al. Specifically for the obsolete test updating task mentioned in the paper. Each sample in the dataset consists of $<original method, updated method, original test, updated test>$. The dataset contains 5,196 samples, which the authors originally split into training and test sets using a 9:1 ratio. We further create a validation set by randomly selecting 11\% of the samples from the training set, resulting in a final split ratio of 8:1:1 for the training, validation and test sets. 