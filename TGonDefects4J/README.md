## Code Structure

```
.
├── README.md                          
├── p1_parse_project                    # Parse the project under test and set up the environment
│   ├── classes.py 
│   ├── clean_env.py
│   ├── code_parser.py
│   ├── extract_focal_method.py
│   ├── merge.py                        # Merge the methods under test
│   └── run.py                          # Includes two functions: 1. Initialize the project under test, 2. Extract the methods under test
├── p2_pred_test_case                   # Perform model inference to generate test cases
│   ├── pred_decoder_tg.py
│   ├── pred_encoder_decoder_tg.py
│   ├── pred_encoder_tg.py
│   ├── pred_open_source_tg.py
│   └── run_example.sh
└── p3_handle_and_run                   # Execute test cases and collect results
    ├── calc.py                         # Calculate multiple test results
    ├── env_check.py
    ├── handle_result.py                # Convert prediction results to executable code
    ├── inject_and_bug_detect.py        # Detect bugs
    ├── inject_and_execute_defect.py    # Inject and execute test cases
    └── single_calculate.py             # Calculate single test result
```
## Detail Information

### Test generation

#### concrate information and generate input content as format below:

`src_fm_fc_ms_ff: <FOCAL_CLASS_NAME> <FOCAL_METHOD> <CONTRSUCTORS> <METHOD_SIGNATURES> <FIELDS>`

example:

`LobbyConfiguration { public int getTeamMaximum() { return maxTeams; } LobbyConfiguration(int maxTeams, Duration intermission, Duration preparation, LobbyScheduleFactory scheduleFactory); int getTeamMaximum(); Duration getIntermission(); Duration getPreparation(); LobbyScheduleFactory getScheduleFactory(); boolean equals(Object other); @Override int hashCode(); @Override String toString();  }`

#### modify generated test case to meet the format requirement

1. add indentations and line breaks
2. add context info, such as: 
    - package info 
    - import info 
    - class struct

### Test execute

#### metric setting (follow methods2test)

1. **Syntax Error** : the test has syntax errors;
2. **Build Error** : the test has correct syntax but fails to build;
3. **Failing Test** : the test builds but fails due to wrong assertions or expected behavior;
4. **Passing Test** : the test builds and passes;
5. **Correct Test** : the test passes and covers the correct focal method;

#### flow control

1. get original generated test code; (need to attach necessary infomation)
2. extract and construct path info
3. construct test class 
4. check syntax error
5. run `mvn clean test` command make sure environmental correctness
6. write test code into test file
6. check compile error (use mvn test-comile)
7. run test and calculate coverage (use clover, need to add parser logic)
8. statistical result and clean test file.
