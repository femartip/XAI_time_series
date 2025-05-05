def binary_search_function_with_bruteforce(f, y, left, right, precision=1e-10, max_iterations=1000,
                                           bruteforce_points=10000):
    """
    Binary search to find x where f(x) is closest to target value y,
    followed by a bruteforce refinement search

    Args:
        f (function): The function to evaluate
        y (float): The target value we want f(x) to be close to
        left (float): Left boundary of search space
        right (float): Right boundary of search space
        precision (float, optional): How precise the search should be. Defaults to 1e-10.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
        bruteforce_points (int, optional): Number of points to check in bruteforce phase. Defaults to 100.

    Returns:
        float: The value x where f(x) is closest to y
    """
    # First, run the binary search to get an approximate solution
    binary_result = binary_search_function(f, y, left, right, precision, max_iterations)

    off_by = abs(f(binary_result)-y)
    # Determine the order of magnitude of the result for scaling the bruteforce search
    if off_by == 0:
        return binary_result
    elif binary_result == 0:
        magnitude = precision
    else:
        magnitude = abs(binary_result)


    # Define a search window based on the magnitude
    # For small values, use a fixed window size to avoid too small search spaces
    window_size = magnitude * off_by
    search_left = binary_result - window_size
    search_right = binary_result + window_size

    # Perform bruteforce search in the refined window
    best_x = binary_result
    best_diff = abs(f(binary_result) - y)

    step_size = (search_right - search_left) / bruteforce_points

    # Brute force search
    for i in range(bruteforce_points + 1):
        x = search_left + i * step_size
        diff = abs(f(x) - y)

        if diff < best_diff:
            best_x = x
            best_diff = diff

    return best_x


def binary_search_function(f, y, left, right, precision=1e-10, max_iterations=100):
    """
    Binary search to find x where f(x) is closest to target value y

    Args:
        f (function): The function to evaluate
        y (float): The target value we want f(x) to be close to
        left (float): Left boundary of search space
        right (float): Right boundary of search space
        precision (float, optional): How precise the search should be. Defaults to 1e-10.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        float: The value x where f(x) is closest to y
    """
    iterations = 0
    best_x = left
    best_diff = abs(f(left) - y)

    while right - left > precision and iterations < max_iterations:
        mid = (left + right) / 2
        mid_value = f(mid)
        mid_diff = abs(mid_value - y)

        # Track the best approximation found so far
        if mid_diff < best_diff:
            best_x = mid
            best_diff = mid_diff

        # If we've found an exact match (within precision), return it
        if mid_diff < precision:
            return mid

        # Decide which half to search next
        # Check values at both ends of each potential interval
        left_mid = (left + mid) / 2
        right_mid = (mid + right) / 2

        left_mid_diff = abs(f(left_mid) - y)
        right_mid_diff = abs(f(right_mid) - y)

        if left_mid_diff < right_mid_diff:
            right = mid
        elif left_mid_diff > right_mid_diff:
            left = mid
        else:
            right_diff = abs(f(right)-y)
            left_diff = abs(f(left)-y)
            if right_diff == right_mid_diff:
                right = mid
            elif left_diff == left_mid_diff:
                left = mid
            else:
                if right_diff < left_diff:
                    left = mid
                else:
                    right = mid



        iterations += 1

    # Final check between left and right to find closest
    left_diff = abs(f(left) - y)
    right_diff = abs(f(right) - y)

    if left_diff < right_diff and left_diff < best_diff:
        return left
    elif right_diff < best_diff:
        return right
    else:
        return best_x


# Example usage
if __name__ == "__main__":
    import math
    import numpy as np
    import random

    # Example with non-monotonic function
    def wavy_function_1(x):
        # Create a new random number generator with x as the seed
        rng = random.Random(x)

        # Generate a random value between 0 and 1
        random_component = rng.random()

        if random_component < 0.1 and 990 <x * x - 4 * x + 3 + 10 * random_component < 1100:
            return 1000
        # A sine wave superimposed on a parabola
        return x * x - 4 * x + 3 + 10 * random_component

    def wavy_function(x):
        # Create a new random number generator with x as the seed
        rng = random.Random(x)

        # Generate a random value between 0 and 1
        random_component = rng.random()

        if random_component < 0.01:
            return 0.8
        # A sine wave superimposed on a parabola
        return random_component /2

    target_y = 1

    L = -10**20
    R = 10**20
    # Compare results with and without bruteforce refinement
    binary_result = binary_search_function(wavy_function, target_y, L, R)
    refined_result = binary_search_function_with_bruteforce(wavy_function, target_y, L, R)

    print(f"Binary search result: x = {binary_result}, f(x) = {wavy_function(binary_result)}")
    print(f"With bruteforce: x = {refined_result}, f(x) = {wavy_function(refined_result)}")

    # Visualization to compare results (optional)
    try:
        import matplotlib.pyplot as plt

        x_vals = np.linspace(binary_result-0.1, refined_result+0.1, 10**4)
        y_vals = [wavy_function(x) for x in x_vals]

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'b-', label='f(x)')
        plt.axhline(y=target_y, color='r', linestyle='--', label=f'Target y={target_y}')
        plt.plot(binary_result, wavy_function(binary_result), 'go', markersize=10, label='Binary Search Result')
        plt.plot(refined_result, wavy_function(refined_result), 'mo', markersize=10, label='Refined Result')
        plt.grid(True)
        plt.legend()
        plt.title('Comparison of Search Results for Non-Monotonic Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.savefig('search_comparison.png')
        print("Visualization saved as 'search_comparison.png'")
    except ImportError:
        print("Matplotlib not available for visualization")