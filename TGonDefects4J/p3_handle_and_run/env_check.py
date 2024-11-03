import os
import subprocess
import time

def run_defects4j_test(root_dir):
    # Traverse all subdirectories in the root directory
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing directory: {subdir_path}")
            # Switch to the subdirectory
            os.chdir(subdir_path)

            try:
                # Execute defects4j test command
                result = subprocess.run(["defects4j", "test"], capture_output=True, text=True)
                
                # Check if the command executed successfully or if there are no failing tests
                if result.returncode != 0:
                    print(f"Error running defects4j test in {subdir_path}:\n{result.stderr}")
                elif "Failing tests: 0" in result.stdout:
                    print(f"Success in {subdir_path}: No failing tests.")
                else:
                    print(f"Check {subdir_path}: Failing tests or other issues may be present.")
                
            except Exception as e:
                print(f"Exception occurred while processing {subdir_path}: {e}")

if __name__ == "__main__":
    root_path = "/efs_data/sy/defect4j/buggy_bac1"
    dirs = ["chart", "cli", "csv", "gson", "lang"]

    for dir in dirs:
        root_dir = os.path.join(root_path, dir)

        # Start timing
        start_time = time.time()

        # Execute the test function
        run_defects4j_test(root_dir)

        # End timing
        end_time = time.time()

        # Calculate and print the time taken
        elapsed_time = end_time - start_time
        print(f"Time taken for {dir}: {elapsed_time:.2f} seconds")
