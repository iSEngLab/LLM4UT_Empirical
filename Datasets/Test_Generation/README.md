# Test Generation dataset

To reduce the repository size, we have not included the $Methods2Test_{filter}$ dataset. 
Instead, we provide a script to process the original Methods2Test dataset.

The Methods2Test dataset can be downloaded via this link: https://github.com/microsoft/methods2test

## Filter Rules

we use following rules to filter Methods2Test dataset:
1. **Length of Tokens.** To ensure data completeness and simplicity, we filter out entries with token lengths that are either too short or too long. Specifically, we retain entries where the length of the input focal method is between 64 and 2048, and the length of the output test case is between 16 and 512. 
2. **Construction.** To normalize the construction of test cases, we only retain entries that begin with the prefix `@Test public void` and do not throw exceptions.
3. **Duplicated Test Cases.** We find half of the focal methods in the dataset include at least two related test cases. To increase the variety of focal methods, we randomly select one pair from each set of test cases associated with the same focal methods.
4. **Filtered repository.** To maintain the quality of the dataset, we filter out any repository with fewer than 50 pairs. Conversely, to ensure a diverse set of test cases that are not dominated by a few repositories, we randomly sample 200 pairs from any repository with more than 200 pairs.

## Script

- data_filter.py filters the dataset according to the specified rules.
- dataset_split.py splits the filtered dataset into training, validation, and test sets by repository.

In total, the training set contains 44,897 entries, the validation set 5,612, and the test set 5,623.

