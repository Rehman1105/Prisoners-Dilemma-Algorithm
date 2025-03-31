import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Constants
NUM_STRATEGIES = 1000  # Number of random strategies to generate
NUM_ROUNDS = 100       # Number of rounds in each Prisoner's Dilemma game
PAYOFF_MATRIX = {      # Payoff matrix for Prisoner's Dilemma
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

# Helper function to simulate a strategy's move with memory depth of 3
def strategy_move(strategy, history):
    """
    Simulates a strategy's move based on the history of the last 3 moves.
    """
    if len(history) < 3:
        # If there's not enough history, cooperate or defect randomly
        return np.random.choice(['C', 'D'])
    else:
        # Get the last 3 moves of both players
        last_3_moves = history[-3:]
        last_3_opponent_moves = [move[1] for move in last_3_moves]  # Opponent's last 3 moves

        if strategy == 'TFT':  # Tit for Tat
            return last_3_opponent_moves[-1]  # Mirror the opponent's last move
        elif strategy == 'ALLD':  # Always Defect
            return 'D'
        elif strategy == 'ALLC':  # Always Cooperate
            return 'C'
        elif strategy == 'RAND':  # Random
            return np.random.choice(['C', 'D'])
        elif strategy == 'TF2T':  # Tit for Two Tat
            if last_3_opponent_moves[-2:] == ['D', 'D']:  # Defect if opponent defected twice in a row
                return 'D'
            else:
                return 'C'
        elif strategy == 'STFT':  # Suspicious Tit for Tat
            if len(history) == 0:
                return 'D'  # Start by defecting
            else:
                return last_3_opponent_moves[-1]  # Mirror the opponent's last move
        else:
            # Custom strategy: cooperate if opponent cooperated in the last 2 out of 3 moves
            coop_count = last_3_opponent_moves.count('C')
            return 'C' if coop_count >= 2 else 'D'

# Helper function to simulate a game between two strategies
def simulate_game(strategy_a, strategy_b, num_rounds):
    """
    Simulates a Prisoner's Dilemma game between two strategies.
    Returns the total payoff for Strategy B against Strategy A.
    """
    history = []
    total_payoff_b = 0

    for _ in range(num_rounds):
        move_a = strategy_move(strategy_a, history)
        move_b = strategy_move(strategy_b, history)
        payoff_a, payoff_b = PAYOFF_MATRIX[(move_a, move_b)]
        total_payoff_b += payoff_b
        history.append((move_a, move_b))

    return total_payoff_b / num_rounds  # Average payoff for Strategy B

# Generate random strategies
def generate_random_strategy():
    """
    Generates a random strategy for the Prisoner's Dilemma.
    """
    strategies = ['TFT', 'ALLD', 'ALLC', 'RAND', 'TF2T', 'STFT']  # Add more strategies
    return np.random.choice(strategies)

# Generate training data
def generate_training_data(num_strategies, num_rounds):
    """
    Generates a dataset of strategy interactions and their outcomes.
    """
    strategies = [generate_random_strategy() for _ in range(num_strategies)]
    data = []

    for i in range(num_strategies):
        strategy_a = strategies[i]
        for j in range(num_strategies):
            if i == j:
                continue  # Skip self-interactions
            strategy_b = strategies[j]
            avg_payoff_b = simulate_game(strategy_a, strategy_b, num_rounds)
            label = 1 if avg_payoff_b >= 3 else 0  # "Good" if payoff >= 3, else "Bad"
            data.append({
                'strategy_a': strategy_a,
                'strategy_b': strategy_b,
                'avg_payoff_b': avg_payoff_b,
                'label': label
            })
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Generated data for {i + 1} out of {num_strategies} strategies...")

    return pd.DataFrame(data)

# Train the machine learning model
def train_model(data):
    """
    Trains a RandomForestClassifier to predict whether Strategy B is "good" or "bad" against Strategy A.
    """
    # Encode strategies as categorical features
    data['strategy_a'] = data['strategy_a'].astype('category').cat.codes
    data['strategy_b'] = data['strategy_b'].astype('category').cat.codes

    # Features and labels
    X = data[['strategy_a', 'strategy_b']]
    y = data['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))
    print("S-Score:", roc_auc_score(y_test, y_pred))

    return model

# Predict whether Strategy B is "good" or "bad" against Strategy A
def predict_strategy(model, strategy_a, strategy_b):
    """
    Predicts whether Strategy B is "good" or "bad" against Strategy A.
    """
    # Encode strategies
    categories = ['TFT', 'ALLD', 'ALLC', 'RAND', 'TF2T', 'STFT']  # Add all strategies
    strategy_a_code = pd.Categorical([strategy_a], categories=categories).codes[0]
    strategy_b_code = pd.Categorical([strategy_b], categories=categories).codes[0]

    # Create input DataFrame
    input_data = pd.DataFrame({
        'strategy_a': [strategy_a_code],
        'strategy_b': [strategy_b_code]
    })

    # Predict
    prediction = model.predict(input_data)
    return "Good" if prediction[0] == 1 else "Bad"

# Main script
if __name__ == "__main__":
    # Step 1: Generate training data
    print("Generating training data...")
    data = generate_training_data(NUM_STRATEGIES, NUM_ROUNDS)

    # Step 2: Train the model
    print("Training the model...")
    model = train_model(data)

    # Step 3: Allow user to input strategies
    print("\nEnter two strategies to predict their interaction.")
    print("Available strategies: TFT, ALLD, ALLC, RAND, TF2T, STFT")
    
    while True:
        strategy_a = input("Enter Strategy A: ").strip().upper()
        strategy_b = input("Enter Strategy B: ").strip().upper()

        # Validate input
        valid_strategies = ['TFT', 'ALLD', 'ALLC', 'RAND', 'TF2T', 'STFT']
        if strategy_a not in valid_strategies or strategy_b not in valid_strategies:
            print("Invalid strategy. Please try again.")
            continue

        # Make prediction
        prediction = predict_strategy(model, strategy_a, strategy_b)
        print(f"Strategy B ({strategy_b}) is predicted to be {prediction} against Strategy A ({strategy_a}).")

        # Ask if the user wants to make another prediction
        another = input("Do you want to make another prediction?: ").strip().lower()
        if another != 'y':
            break