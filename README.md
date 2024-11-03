# A Large-scale Empirical Study on Fine-tuning Large Language Models for Unit Testing

This repository contains the replication package for the paper **"A Large-scale Empirical Study on Fine-tuning Large Language Models for Unit Testing"**. The study We conduct the first large-scale empirical study on fine-tuning 37 popular
LLMs in three unit testing scenarios across five benchmarks and eight metrics, utilizing over 3,000 NVIDIA A100 GPU hours. 


## Abstract

Unit testing plays a pivotal role in software development, improving software quality and reliability. However, generating effective test cases manually is time-consuming, prompting interest in unit testing research. 
Recently, Large Language Models (LLMs) have shown potential in various unit testing tasks, including test generation, assertion generation, and test evolution, but existing studies are limited in scope and lack a systematic evaluation of the effectiveness of LLMs. 

To bridge this gap, we present a large-scale empirical study on fine-tuning LLMs for unit testing.
Our study involves three unit testing tasks, five benchmarks, eight evaluation metrics, and 37 popular LLMs across various architectures and sizes, consuming over 3,000 NVIDIA A100 GPU hours. 
We focus on three key research questions: (1) the performance of LLMs compared to state-of-the-art methods, (2) the impact of different factors on LLM performance, and (3) the effectiveness of fine-tuning versus prompt engineering. 
Our findings reveal that LLMs outperform existing state-of-the-art approaches on all three unit testing tasks across nearly all metrics, highlighting the potential of fine-tuning LLMs in unit testing tasks. 
Furthermore, large-scale, decoder-only models achieve the best results across tasks, while encoder-decoder models perform better under the same parameter scale. 
Additionally, the comparison of the performance between fine-tuning and prompt engineering approaches reveals the considerable potential capability of the prompt engineering approach in unit testing tasks. 
We then discuss the concerned issues on the test generation task, including data leakage issues, bug detection capabilities, and metrics comparisons. 
Finally, we further pinpoint carious practical guidelines for LLM-based approaches to unit testing tasks in the near future.
Overall, our work demonstrates the promising future of fine-tuning LLMs on unit testing tasks and reduces the manual efforts of unit testing experts in practical scenarios.

## Project Structure

The repository contains:
- Datasets: the fine-tuning dataset for three unit testing tasks
- Finetune_Script: the scripts for LLM finetuning
- Inference_Script: the scripts for prompt engineering approach
- TextEvaluate: the evaluate script for text-based metrics
- TGonDefects4J: the evaluate script for runtime-based metrics on Defects4J dataset.
- **Prompt_template.md**: the template of prompt using for prompt engineering approach.
- **Total_result.pdf**: the complete result of LLMs on three unit testing tasks.

## Research Questions
1. **RQ1:** How does the performance of fine-tuning LLMs compare to existing approaches on three unit testing tasks?
2. **RQ2:** What is the impact of various factors (including model series, model architecture, and model size) on the performance of LLMs?
3. **RQ3:** How do fine-tuning approaches perform compared to prompt engineering approaches?

## How to Use
1. **Clone the Repository:**
```
git clone https://anonymous.4open.science/r/LLM4UT-ISSTA
cd LLM-Unit-Test-Generation
```
2. **Setup Dependencies:** 
Ensure all necessary dependencies are available, especially fine-tune dataset and [Defects4J](https://github.com/rjust/defects4j) dataset.
3. **Run Experiments:**
Navigate to the `Finetune_Script` and `Inference_Script` directory and follow the instructions in the respective subdirectories to reproduce the experiments.
4. **Evaluate Results:**
Navigate to the `TextEvaluate` and `TGonDefects4J` directory and follow the instructions in the respective subdirectories to evaluate the results.

## Citation
If you use this package in your research, please cite our paper:

```
@article{Paper,
  title={A Large-scale Empirical Study on Fine-tuning Large Language Models for Unit Testing},
  author={Anonymous},
  journal={Journal},
  year={2024}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the developers of the tools and datasets used in this study. Special thanks to the reviewers for their valuable feedback.