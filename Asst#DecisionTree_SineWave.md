# Decision Tree Regression on Sine Wave Function

## Overview
This test evaluates the decision tree's ability to approximate a continuous sine wave function from 0 to 2π, demonstrating its regression capabilities on a smooth, non-linear function.

## Experimental Setup

### Dataset Generation
- **Function**: y = sin(x) where x ∈ [0, 2π]
- **Training Points**: 100 samples uniformly distributed
- **Test Points**: 300 samples (higher resolution for smooth visualization)
- **Noise Level**: Gaussian noise with σ = 0.1 added to training data
- **Target**: Predict clean sine wave values

### Model Configuration
- **Algorithm**: Decision Tree Regression
- **Criterion**: Information Gain (MSE-based)
- **Depths Tested**: [3, 5, 7, 10, 15]
- **Evaluation**: RMSE and MAE on clean sine wave

## Results Summary

### Performance by Depth

| Max Depth | RMSE   | MAE    | Performance |
|-----------|--------|--------|-------------|
| 3         | 0.1653 | 0.1380 | Poor - too simple |
| 5         | 0.0790 | 0.0650 | Good improvement |
| **7**     | **0.0715** | **0.0544** | **Best performance** |
| 10        | 0.0886 | 0.0699 | Overfitting starts |
| 15        | 0.0924 | 0.0727 | Clear overfitting |

### Key Findings

1. **Optimal Depth**: 7 provides the best balance between approximation quality and overfitting
2. **Overfitting Pattern**: Performance degrades beyond depth 7 due to fitting training noise
3. **Approximation Quality**: Best RMSE of 0.0715 indicates good sine wave approximation
4. **Piecewise Nature**: Decision trees create step-like approximations of smooth functions

## Noise Sensitivity Analysis

### Performance vs Noise Level (Depth = 10)

| Noise Level | RMSE   | Impact |
|-------------|--------|--------|
| 0.0         | 0.0135 | Excellent - nearly perfect |
| 0.05        | 0.0537 | Good - minimal degradation |
| 0.1         | 0.0898 | Moderate - noticeable impact |
| 0.2         | 0.2095 | Poor - significant degradation |
| 0.3         | 0.2707 | Very poor - heavy noise impact |

### Observations
- **Clean Data**: Nearly perfect approximation (RMSE = 0.0135)
- **Low Noise**: Robust performance up to σ = 0.05
- **High Noise**: Significant performance degradation beyond σ = 0.1
- **Noise Sensitivity**: Decision trees are moderately sensitive to training noise

## Visual Analysis

### Approximation Characteristics
1. **Step Function Nature**: Decision trees create piecewise constant predictions
2. **Segment Quality**: Higher depth = more segments = better local approximation
3. **Smooth Function Challenge**: Inherent limitation for continuous functions
4. **Boundary Effects**: Sharp transitions at decision boundaries

### Tree Structure Insights
The optimal tree (depth 7) shows:
- **Primary Split**: x ≤ 3.078 (approximately π)
- **Hierarchical Refinement**: Progressive subdivision of x-axis
- **Local Averaging**: Each leaf represents local mean of training points
- **Complexity**: 128 potential leaf nodes (2^7) with actual usage varying

## Theoretical Analysis

### Decision Tree Limitations for Sine Waves
1. **Discontinuous Approximation**: Cannot represent smooth curves directly
2. **Axis-Aligned Splits**: Only vertical splits in 1D, creating steps
3. **Local Constant Prediction**: Each region has constant output
4. **Overfitting Tendency**: Deep trees memorize noise rather than pattern

### Comparison with Ideal Methods
- **Polynomial Regression**: Would better capture smooth nature
- **Spline Interpolation**: Natural choice for smooth function approximation
- **Neural Networks**: Could learn smooth approximations
- **Fourier Series**: Ideal for periodic functions like sine waves

## Practical Implications

### When Decision Trees Work Well for Regression
1. **Piecewise Constant Functions**: Natural fit
2. **Feature Interactions**: Good at capturing complex interactions
3. **Non-parametric**: No assumptions about function form
4. **Interpretability**: Clear decision rules

### When Decision Trees Struggle
1. **Smooth Functions**: Like sine waves, exponentials
2. **Linear Relationships**: Simple linear regression better
3. **High-Dimensional Smooth Manifolds**: Curse of dimensionality
4. **Extrapolation**: Poor performance outside training range

## Conclusions

### Performance Assessment
1. **Reasonable Approximation**: RMSE of 0.0715 shows decent performance
2. **Optimal Depth Found**: Depth 7 balances bias-variance tradeoff
3. **Noise Robustness**: Moderate sensitivity to training noise
4. **Characteristic Limitations**: Step-like nature inherent to method

### Recommendations
1. **Use Case**: Decision trees suitable for rough sine wave approximation
2. **Preprocessing**: Consider feature engineering (e.g., x², x³ terms)
3. **Ensemble Methods**: Random Forest might improve smoothness
4. **Alternative Methods**: Consider splines or neural networks for smooth functions

### Educational Value
This test demonstrates:
- **Algorithm Behavior**: How decision trees handle continuous functions
- **Bias-Variance Tradeoff**: Optimal depth selection
- **Method Limitations**: Understanding when not to use decision trees
- **Performance Evaluation**: Proper regression assessment techniques

## Files Generated
- `sine_wave_decision_tree_results.png`: Predictions vs true sine wave for different depths
- `sine_wave_performance_comparison.png`: RMSE and MAE vs depth curves
- `sine_wave_noise_sensitivity.png`: Performance under different noise levels

The test successfully demonstrates both the capabilities and limitations of decision tree regression on smooth, continuous functions.
