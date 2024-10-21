import os
import subprocess

# Define ranges for k and numiter
k_values = list(range(5, 101, 1))  # k values starting from 5, 10, 15, ..., up to 100
numiter_values = list(range(50, 501, 50))  # numiter values starting from 50, 100, 150, ..., up to 500

# Output file for results
output_file = "results.csv"

# Clear the output file and add headers before starting
with open(output_file, 'w') as f:
    f.write("k,numiter,acc_pos,acc_neg,avg_accuracy\n")

# Function to run bow.py with specific hyperparameters
def run_experiment(k, numiter, output_file):
    print(f"Running experiment with k={k} and numiter={numiter}...")
    subprocess.run(['python', 'bow.py', '--k', str(k), '--numiter', str(numiter), '--output', output_file])

# Loop over k and numiter values and run experiments
for k in k_values:
    for numiter in numiter_values:
        run_experiment(k, numiter, output_file)

print(f"All experiments finished. Results saved to {output_file}")
