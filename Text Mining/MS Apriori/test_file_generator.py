import random

# number of transactions
min_rows = 50
max_rows = 100

# numebr of items
min_N = 30
max_N = 40

# number of items per transaction
min_M = 1
max_M = min_N

# percentage of items with their own MIS value
min_K = 0.6
max_K = 0.9

# MIS values for each item
min_p = 0.25
max_p = 0.75

# SDC (phi) value for the itemsets
min_SDC = 0.1
max_SDC = 0.3

# Generate data.txt
def generate_transaction_file(filename="data.txt"):
    num_rows = random.randint(min_rows, max_rows)

    with open(filename, 'w') as file:
        for _ in range(num_rows):
            M = random.randint(min_M, max_M)
            transaction = random.sample(range(1, N + 1), M)  # Generate M unique numbers from 1 to N
            transaction_str = ', '.join(map(str, transaction))
            file.write(f"{transaction_str}\n")

# Generate params.txt
def generate_params_file(filename="parameters.txt"):
    K = random.uniform(min_K, max_K)
    max_index = int(N * K)
    
    with open(filename, 'w') as file:
        # Generate MIS values for each number in range 1 to max_index
        for i in range(1, max_index + 1):
            p = round(random.uniform(min_p, max_p), 2)
            file.write(f"MIS({i}) = {p}\n")
        
        # Generate MIS(rest) value
        p_rest = round(random.uniform(min_p, max_p), 2)
        file.write(f"MIS(rest) = {p_rest}\n")
        
        # Generate SDC value
        t = round(random.uniform(min_SDC, max_SDC), 2)
        file.write(f"SDC = {t}\n")

N = random.randint(min_N, max_N)
# Generate both files
generate_transaction_file()
generate_params_file()