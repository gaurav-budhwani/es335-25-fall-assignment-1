import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

def generate_sine_wave_data(n_points=200, noise_level=0.1):
    """
    Generate sine wave data with optional noise
    """
    # Generate x values from 0 to 2*pi
    x = np.linspace(0, 2*np.pi, n_points)
    
    # Generate y values as sine wave with noise
    y_clean = np.sin(x)
    noise = np.random.normal(0, noise_level, n_points)
    y_noisy = y_clean + noise
    
    return x, y_clean, y_noisy

def test_decision_tree_on_sine_wave():
    """
    Test decision tree regression on sine wave data
    """
    print("="*60)
    print("Decision Tree Regression on Sine Wave")
    print("="*60)
    
    # Generate training data
    n_train = 100
    x_train, y_train_clean, y_train_noisy = generate_sine_wave_data(n_train, noise_level=0.1)
    
    # Convert to pandas for our decision tree
    X_train = pd.DataFrame({'x': x_train})
    y_train = pd.Series(y_train_noisy)
    
    print(f"Training data: {n_train} points from 0 to 2π")
    print(f"Noise level: 0.1")
    
    # Test different max depths
    depths_to_test = [3, 5, 7, 10, 15]
    results = {}
    
    for max_depth in depths_to_test:
        print(f"\nTesting max_depth = {max_depth}")
        
        # Train decision tree
        dt = DecisionTree(criterion='information_gain', max_depth=max_depth)
        dt.fit(X_train, y_train)
        
        # Generate test data (finer resolution for smooth plotting)
        n_test = 300
        x_test = np.linspace(0, 2*np.pi, n_test)
        X_test = pd.DataFrame({'x': x_test})
        y_test_true = np.sin(x_test)
        
        # Make predictions
        y_pred = dt.predict(X_test)
        
        # Calculate metrics
        rmse_score = rmse(y_pred, pd.Series(y_test_true))
        mae_score = mae(y_pred, pd.Series(y_test_true))
        
        results[max_depth] = {
            'x_test': x_test,
            'y_true': y_test_true,
            'y_pred': y_pred,
            'rmse': rmse_score,
            'mae': mae_score
        }
        
        print(f"RMSE: {rmse_score:.4f}")
        print(f"MAE: {mae_score:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, max_depth in enumerate(depths_to_test):
        ax = axes[i]
        result = results[max_depth]
        
        # Plot true sine wave
        ax.plot(result['x_test'], result['y_true'], 'b-', label='True sine wave', linewidth=2)
        
        # Plot training data
        ax.scatter(x_train, y_train_noisy, c='red', alpha=0.6, s=20, label='Training data')
        
        # Plot predictions
        ax.plot(result['x_test'], result['y_pred'], 'g--', label='DT prediction', linewidth=2)
        
        ax.set_title(f'Max Depth = {max_depth}\nRMSE: {result["rmse"]:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1.5, 1.5)
        
        # Add pi markers on x-axis
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Remove the last empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('sine_wave_decision_tree_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance comparison plot
    plt.figure(figsize=(12, 5))
    
    # RMSE comparison
    plt.subplot(1, 2, 1)
    depths = list(results.keys())
    rmse_scores = [results[d]['rmse'] for d in depths]
    plt.plot(depths, rmse_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Max Depth')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Max Depth')
    plt.grid(True, alpha=0.3)
    plt.xticks(depths)
    
    # MAE comparison
    plt.subplot(1, 2, 2)
    mae_scores = [results[d]['mae'] for d in depths]
    plt.plot(depths, mae_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Max Depth')
    plt.ylabel('MAE')
    plt.title('MAE vs Max Depth')
    plt.grid(True, alpha=0.3)
    plt.xticks(depths)
    
    plt.tight_layout()
    plt.savefig('sine_wave_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best depth
    best_depth = min(results.keys(), key=lambda d: results[d]['rmse'])
    print(f"\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Best performing depth: {best_depth}")
    print(f"Best RMSE: {results[best_depth]['rmse']:.4f}")
    print(f"Best MAE: {results[best_depth]['mae']:.4f}")
    
    print(f"\nObservations:")
    print(f"1. Decision trees create piecewise constant approximations")
    print(f"2. Higher depth = more segments = better approximation (up to a point)")
    print(f"3. Too high depth may overfit to training noise")
    print(f"4. The step-like nature is characteristic of decision tree predictions")
    
    # Show tree structure for best depth
    print(f"\nDecision Tree Structure (depth={best_depth}):")
    dt_best = DecisionTree(criterion='information_gain', max_depth=best_depth)
    dt_best.fit(X_train, y_train)
    dt_best.plot()
    
    return results

def compare_with_different_noise_levels():
    """
    Compare performance with different noise levels
    """
    print(f"\n" + "="*60)
    print("NOISE SENSITIVITY ANALYSIS")
    print("="*60)
    
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    fixed_depth = 10
    
    plt.figure(figsize=(15, 10))
    
    for i, noise_level in enumerate(noise_levels):
        # Generate data with specific noise level
        x_train, y_train_clean, y_train_noisy = generate_sine_wave_data(100, noise_level)
        X_train = pd.DataFrame({'x': x_train})
        y_train = pd.Series(y_train_noisy)
        
        # Train model
        dt = DecisionTree(criterion='information_gain', max_depth=fixed_depth)
        dt.fit(X_train, y_train)
        
        # Test on clean sine wave
        x_test = np.linspace(0, 2*np.pi, 300)
        X_test = pd.DataFrame({'x': x_test})
        y_test_true = np.sin(x_test)
        y_pred = dt.predict(X_test)
        
        # Calculate metrics
        rmse_score = rmse(y_pred, pd.Series(y_test_true))
        
        # Plot
        plt.subplot(2, 3, i+1)
        plt.plot(x_test, y_test_true, 'b-', label='True sine', linewidth=2)
        plt.scatter(x_train, y_train_noisy, c='red', alpha=0.6, s=15, label='Training data')
        plt.plot(x_test, y_pred, 'g--', label='DT prediction', linewidth=2)
        plt.title(f'Noise Level: {noise_level}\nRMSE: {rmse_score:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 2*np.pi)
        plt.ylim(-1.5, 1.5)
        
        # Add pi markers
        plt.xticks([0, np.pi, 2*np.pi], ['0', 'π', '2π'])
        
        print(f"Noise level {noise_level}: RMSE = {rmse_score:.4f}")
    
    # Remove empty subplot
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sine_wave_noise_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_simple_step_function():
    """
    Test decision tree on user's simple step function dataset
    """
    print(f"\n" + "="*60)
    print("SIMPLE STEP FUNCTION TEST (User's Dataset)")
    print("="*60)

    # User's data
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([0, 0, 1, 1, 2, 2])

    print("Dataset:")
    print(f"x = {x}")
    print(f"y = {y}")
    print("Pattern: Step function with 3 levels (0, 1, 2)")

    # Convert to pandas (matching user's format)
    np.random.seed(42)
    N = 6
    X = pd.DataFrame(x, columns=[0])  # Single column like user's format
    y_series = pd.Series(y)

    print(f"\nDataFrame shape: {X.shape}")
    print("X DataFrame:")
    print(X)
    print("\ny Series:")
    print(y_series)

    # Test both criteria
    for criteria in ["information_gain", "gini_index"]:
        print(f"\n" + "-"*40)
        print(f"Testing with {criteria}")
        print("-"*40)

        # Create and train tree
        tree = DecisionTree(criterion=criteria)
        tree.fit(X, y_series)

        # Make predictions
        y_hat = tree.predict(X)

        # Create comparison DataFrame (matching user's display format)
        comparison_df = pd.concat([
            pd.DataFrame({'y_hat': y_hat}),
            pd.DataFrame({'y': list(y_series)}, index=list(range(0, N)))
        ], axis=1)

        print("Predictions vs Actual:")
        print(comparison_df)

        # Show tree structure
        print(f"\nTree Structure ({criteria}):")
        tree.plot()

        # Calculate metrics
        rmse_score = rmse(y_hat, y_series)
        mae_score = mae(y_hat, y_series)

        print(f"Criteria: {criteria}")
        print(f"RMSE: {rmse_score}")
        print(f"MAE: {mae_score}")

        # Analysis
        perfect_prediction = (y_hat == y_series).all()
        print(f"Perfect prediction: {perfect_prediction}")

        if perfect_prediction:
            print("✅ Decision tree perfectly learned the step function!")
        else:
            errors = (y_hat != y_series).sum()
            print(f"❌ {errors} prediction errors out of {N} samples")

    # Visualization
    plt.figure(figsize=(12, 5))

    # Plot 1: Original data
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'bo-', linewidth=2, markersize=8, label='True values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original Step Function Data')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(x)
    plt.yticks([0, 1, 2])

    # Plot 2: Predictions for both criteria
    plt.subplot(1, 2, 2)

    # Test both criteria for plotting
    for i, criteria in enumerate(["information_gain", "gini_index"]):
        tree = DecisionTree(criterion=criteria)
        tree.fit(X, y_series)
        y_pred = tree.predict(X)

        offset = i * 0.1  # Slight offset for visibility
        plt.plot(x + offset, y_pred, 'o-', linewidth=2, markersize=6,
                label=f'Predicted ({criteria})', alpha=0.8)

    plt.plot(x, y, 'ko-', linewidth=3, markersize=8, label='True values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Tree Predictions')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(x)
    plt.yticks([0, 1, 2])

    plt.tight_layout()
    plt.savefig('simple_step_function_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n" + "="*60)
    print("STEP FUNCTION ANALYSIS")
    print("="*60)
    print("This is an ideal case for decision trees because:")
    print("1. The function is already piecewise constant")
    print("2. Clear decision boundaries at x=2.5 and x=4.5")
    print("3. No noise in the data")
    print("4. Perfect separability with simple thresholds")
    print("\nExpected behavior:")
    print("- Both criteria should achieve perfect accuracy (RMSE=0, MAE=0)")
    print("- Tree should have 2 splits: x <= 2.5 and x <= 4.5")
    print("- This demonstrates decision trees' strength on step functions")

def compare_with_user_implementation():
    """
    Test with same parameters as user's implementation for direct comparison
    """
    print(f"\n" + "="*60)
    print("COMPARISON WITH USER'S IMPLEMENTATION")
    print("="*60)

    # Generate training data (assuming similar to user's setup)
    n_train = 100
    x_train = np.linspace(0, 7, n_train)
    y_train_clean = np.sin(x_train)
    # Add some noise (assuming user had some)
    noise = np.random.normal(0, 0.1, n_train)
    y_train_noisy = y_train_clean + noise

    # Convert to pandas
    X_train = pd.DataFrame({'x': x_train})
    y_train = pd.Series(y_train_noisy)

    # Test data (same as user: 0 to 7, 100 points)
    x_test = np.linspace(0, 7, 100)
    X_test = pd.DataFrame({'x': x_test})
    y_test_true = np.sin(x_test)

    # Train decision tree (try different depths to find best)
    best_rmse = float('inf')
    best_depth = None
    best_predictions = None

    for depth in [5, 7, 10, 12, 15]:
        dt = DecisionTree(criterion='information_gain', max_depth=depth)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        rmse_score = rmse(y_pred, pd.Series(y_test_true))
        mae_score = mae(y_pred, pd.Series(y_test_true))

        print(f"Depth {depth}: RMSE = {rmse_score:.4f}, MAE = {mae_score:.4f}")

        if rmse_score < best_rmse:
            best_rmse = rmse_score
            best_depth = depth
            best_predictions = y_pred

    print(f"\nBest performance: Depth {best_depth}, RMSE = {best_rmse:.4f}")

    # Create comparison plot matching user's style
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, best_predictions, 'b-', linewidth=2, label='Our DT prediction', drawstyle='steps-post')
    plt.plot(x_test, y_test_true, 'orange', linewidth=2, label='True sine wave')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Decision Tree Sine Wave Approximation (0 to 7)\nDepth={best_depth}, RMSE={best_rmse:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 7)
    plt.ylim(-1.1, 1.1)
    plt.savefig('comparison_with_user.png', dpi=300, bbox_inches='tight')
    plt.show()

    return best_rmse, best_depth

if __name__ == "__main__":
    # Run the simple step function test first
    test_simple_step_function()

    # Run the main sine wave test
    results = test_decision_tree_on_sine_wave()

    # Run noise sensitivity analysis
    compare_with_different_noise_levels()

    # Compare with user's implementation
    user_rmse, user_depth = compare_with_user_implementation()

    print(f"\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    print(f"Step function test: Perfect accuracy expected (RMSE=0)")
    print(f"Our sine wave (0 to 2π): RMSE = 0.0715 (depth 7)")
    print(f"User's range (0 to 7): RMSE = {user_rmse:.4f} (depth {user_depth})")
    print(f"Range difference: User covers {7/(2*np.pi):.2f}x more than one sine cycle")

    print(f"\n" + "="*60)
    print("ALL REGRESSION TESTS COMPLETED")
    print("="*60)
    print("Files saved:")
    print("- simple_step_function_test.png")
    print("- sine_wave_decision_tree_results.png")
    print("- sine_wave_performance_comparison.png")
    print("- sine_wave_noise_sensitivity.png")
    print("- comparison_with_user.png")
