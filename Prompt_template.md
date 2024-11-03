# Prompt Template

## Assertion Generation

```
Given the following Java test case, complete the <AssertPlaceHolder> with an appropriate assertion.
Your task is to generate the correct assertion statement that should replace the <AssertPlaceHolder>.
The output should only include the generated assertion statement, wrapped in a Java code cell.

## Test Case
{source}
```

## Test Generation

```
Please write JUnit 3 unit test for function: `{signature}`, 
output completed test code in a java code cell, the full content is given below.
{source}
```

## Test Evolution

```
Please write the updated test case based on original method, updated method and original test case. 
Output updated test case in a java code cell.

## Original Method
{focal_src}
## Updated Method
{focal_tgt}
## Original Test Case
{test_src}
```