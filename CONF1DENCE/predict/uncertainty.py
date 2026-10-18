"""
Uncertainty quantification for stellar evolution tracks and isochrones.

This module provides tools for:
- Generating tracks/isochrones with uncertainty estimates
- Finding confidence regions for reliable predictions
- Adaptive sampling in HR diagram space
"""

import numpy as np
import pandas as pd
import h5py
import torch
from scipy.ndimage import label as connected_components
from ..utils.helper import data_transform as dt


# =============================================================================
# Helper Functions (Private)
# =============================================================================

def _create_label_array(FeH, He_abundance, AlphaFe, age_or_mass_values, 
                        SEM_format, mode='track', fixed_mass=1.0, fixed_age=1.0e10):
    """
    Create label array for emulator input.
    
    Parameters
    ----------
    FeH : float
        Metallicity [Fe/H]
    He_abundance : float
        Helium abundance (Y)
    AlphaFe : float
        Alpha enhancement [α/Fe]
    age_or_mass_values : array-like
        Ages (for track mode) or masses (for isochrone mode)
    SEM_format : dict
        Format dictionary containing label defaults
    mode : str, optional
        'track' or 'isochrone' (default: 'track')
    fixed_mass : float, optional
        Mass for track mode (default: 1.0)
    fixed_age : float, optional
        Age for isochrone mode (default: 1e10)
    
    Returns
    -------
    labels : ndarray
        Shape (n_points, n_features) with all stellar parameters
    """
    age_or_mass_values = np.atleast_1d(age_or_mass_values)
    n_points = len(age_or_mass_values)
    
    # Start with default parameters (exclude age and mass)
    base_labels = np.array(SEM_format['label_default'][:-2])
    
    # Update user-specified parameters
    base_labels[0] = FeH
    base_labels[1] = He_abundance
    base_labels[2] = AlphaFe
    
    # Repeat for all points
    labels = np.tile(base_labels, (n_points, 1))
    
    # Add age and mass depending on mode
    if mode == 'track':
        ages = age_or_mass_values.reshape(-1, 1)
        masses = np.full((n_points, 1), fixed_mass)
    else:  # isochrone
        ages = np.full((n_points, 1), fixed_age)
        masses = age_or_mass_values.reshape(-1, 1)
    
    labels = np.hstack([labels, ages, masses])
    return labels


def _vary_labels_with_uncertainty(base_labels, n_samples, SEM_format, seed=42):
    """
    Generate n_samples variations of labels according to MC distributions.
    
    Parameters
    ----------
    base_labels : ndarray
        Shape (n_features,) - base parameter values
    n_samples : int
        Number of samples to generate
    SEM_format : dict
        Format dictionary with label_MC_params and label_MC_scaler
    seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    varied_labels : ndarray
        Shape (n_samples, n_features) with variations applied
    """
    np.random.seed(seed)
    n_features = len(base_labels)
    varied_labels = np.tile(base_labels, (n_samples, 1))
    
    mc_params = SEM_format['label_MC_params']
    mc_scalers = SEM_format['label_MC_scaler']
    
    for i in range(n_features):
        scaler = mc_scalers[i]
        params = mc_params[i]
        
        if scaler == "None" or params == "":
            continue
            
        elif scaler == "linear":
            # Uniform distribution with reflection at boundaries
            min_val, max_val = params
            samples = np.random.uniform(min_val, max_val, n_samples)
            
            # Reflect samples that exceed boundaries
            while np.any((samples < min_val) | (samples > max_val)):
                below = samples < min_val
                above = samples > max_val
                samples[below] = 2 * min_val - samples[below]
                samples[above] = 2 * max_val - samples[above]
            
            varied_labels[:, i] = samples
            
        elif scaler == "normal":
            # Gaussian distribution (multiplicative factor)
            mean, std = params
            factors = np.random.normal(mean, std, n_samples)
            varied_labels[:, i] = base_labels[i] * factors
            
        elif scaler == "binary":
            # 50-50 choice between two values
            val1, val2 = params
            choices = np.random.choice([val1, val2], size=n_samples, p=[0.5, 0.5])
            varied_labels[:, i] = choices
    
    return varied_labels


def _calculate_mad_uncertainty(samples):
    """
    Calculate median and MAD-based standard deviation.
    
    Parameters
    ----------
    samples : torch.Tensor or ndarray
        Shape (n_samples, n_features)
    
    Returns
    -------
    median : ndarray
        Median values
    std : ndarray
        Standard deviation estimated from MAD (1.4826 * MAD)
    """
    if torch.is_tensor(samples):
        samples = samples.cpu().detach().numpy()
    
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median), axis=0)
    std = 1.4826 * mad  # Convert MAD to std for normal distribution
    
    return median, std


def _test_confidence_at_point(emulator, FeH, He_abundance, AlphaFe, test_value, 
                               SEM_format, mode='track', fixed_mass=1.0, fixed_age=1.0e10,
                               n_samples=1000, batch_size=100000, mad_thresholds=None):
    """
    Test if a single point is confident based on MAD thresholds.
    
    Returns
    -------
    is_confident : bool
        Whether the point passes all MAD thresholds
    mad_dict : dict
        MAD values for each output parameter
    """
    if mad_thresholds is None:
        mad_thresholds = {}
    
    labels = _create_label_array(FeH, He_abundance, AlphaFe, np.array([test_value]),
                                  SEM_format, mode, fixed_mass, fixed_age)
    labels_repeated = np.repeat(labels, n_samples, axis=0)
    norm_labels = dt.norm_data(labels_repeated, SEM_format['label_scaler'], 
                               SEM_format['label_scale'])
    norm_labels_tensor = torch.tensor(norm_labels, dtype=torch.float32).to(emulator.device)
    
    # Process in batches
    all_samples = []
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        with torch.no_grad():
            batch_samples = emulator.gen_theory_models(norm_labels_tensor[batch_start:batch_end])
        all_samples.append(batch_samples)
    
    samples = torch.cat(all_samples, dim=0)
    median, std = _calculate_mad_uncertainty(samples)
    
    is_confident = True
    mad_dict = {}
    for j, header in enumerate(SEM_format['data_header']):
        mad_dict[header] = std[j]
        if header in mad_thresholds and std[j] >= mad_thresholds[header]:
            is_confident = False
    
    return is_confident, mad_dict


def _binary_search_boundary(emulator, FeH, He_abundance, AlphaFe, SEM_format,
                            mode, fixed_mass, fixed_age, start_point, 
                            search_min, search_max, direction, tolerance,
                            n_samples, batch_size, mad_thresholds, max_iterations=20):
    """
    Binary search to find confidence boundary.
    
    Parameters
    ----------
    direction : str
        'upper' or 'lower'
    tolerance : float
        Relative tolerance for convergence
    
    Returns
    -------
    boundary : float
        The boundary value
    iterations : int
        Number of iterations used
    """
    if direction == 'upper':
        low, high = start_point, search_max
        is_conf_high, _ = _test_confidence_at_point(
            emulator, FeH, He_abundance, AlphaFe, high, SEM_format,
            mode, fixed_mass, fixed_age, n_samples, batch_size, mad_thresholds
        )
        if is_conf_high:
            print(f"  Upper boundary beyond {high:.3e}, using as limit")
            return high, 0
        
        for iteration in range(max_iterations):
            mid = np.exp((np.log(low) + np.log(high)) / 2)
            is_confident, _ = _test_confidence_at_point(
                emulator, FeH, He_abundance, AlphaFe, mid, SEM_format,
                mode, fixed_mass, fixed_age, n_samples, batch_size, mad_thresholds
            )
            if is_confident:
                low = mid
            else:
                high = mid
            
            if abs(high - low) / low < tolerance:
                print(f"  Upper boundary converged at {low:.3e} after {iteration+1} iterations")
                return low, iteration + 1
        
        print(f"  Upper boundary: {low:.3e} (max iterations)")
        return low, max_iterations
    
    else:  # lower
        low, high = search_min, start_point
        is_conf_low, _ = _test_confidence_at_point(
            emulator, FeH, He_abundance, AlphaFe, low, SEM_format,
            mode, fixed_mass, fixed_age, n_samples, batch_size, mad_thresholds
        )
        if is_conf_low:
            print(f"  Lower boundary beyond {low:.3e}, using as limit")
            return low, 0
        
        for iteration in range(max_iterations):
            mid = np.exp((np.log(low) + np.log(high)) / 2)
            is_confident, _ = _test_confidence_at_point(
                emulator, FeH, He_abundance, AlphaFe, mid, SEM_format,
                mode, fixed_mass, fixed_age, n_samples, batch_size, mad_thresholds
            )
            if is_confident:
                high = mid
            else:
                low = mid
            
            if abs(high - low) / high < tolerance:
                print(f"  Lower boundary converged at {high:.3e} after {iteration+1} iterations")
                return high, iteration + 1
        
        print(f"  Lower boundary: {high:.3e} (max iterations)")
        return high, max_iterations


def _calculate_hr_arc_lengths(log_t_values, log_l_values):
    """Calculate arc lengths between consecutive points in HR diagram."""
    delta_t = np.diff(log_t_values)
    delta_l = np.diff(log_l_values)
    arc_lengths = np.sqrt(delta_t**2 + delta_l**2)
    return arc_lengths


def _find_largest_gap_index(arc_lengths):
    """Find index of the largest gap in HR diagram."""
    return np.argmax(arc_lengths)


def _interpolate_value_for_gap(values, gap_index):
    """
    Interpolate value (age or mass) to bisect the gap in HR diagram.
    Uses geometric mean for logarithmic quantities.
    """
    val_before = values[gap_index]
    val_after = values[gap_index + 1]
    return np.exp((np.log(val_before) + np.log(val_after)) / 2)


# =============================================================================
# Public API
# =============================================================================

def find_confidence_region(emulator, FeH, He_abundance, AlphaFe, SEM_format,
                           mode='track', fixed_mass=1.0, fixed_age=1.0e10,
                           n_samples=1000, batch_size=100000,
                           mad_thresholds=None, boundary_tolerance=0.1,
                           visualize=False, n_viz_points=1000):
    """
    Find confidence region boundaries using adaptive binary search.
    
    Parameters
    ----------
    emulator : CONF1DENCE.predict.emulator
        Trained emulator object
    FeH : float
        Metallicity [Fe/H]
    He_abundance : float
        Helium abundance (Y)
    AlphaFe : float
        Alpha enhancement [α/Fe]
    SEM_format : dict
        Format dictionary from emulator
    mode : str, optional
        'track' or 'isochrone' (default: 'track')
    fixed_mass : float, optional
        Mass for track mode in solar masses (default: 1.0)
    fixed_age : float, optional
        Age for isochrone mode in years (default: 1e10)
    n_samples : int, optional
        Number of samples per test point (default: 1000)
    batch_size : int, optional
        Batch size for GPU processing (default: 100000)
    mad_thresholds : dict, optional
        MAD thresholds for each parameter (default: reasonable values)
    boundary_tolerance : float, optional
        Relative tolerance for boundary convergence (default: 0.1)
    visualize : bool, optional
        If True, perform full mapping for visualization (default: False)
    n_viz_points : int, optional
        Number of points for visualization if enabled (default: 1000)
    
    Returns
    -------
    If visualize=False:
        min_confident : float
            Lower boundary of confident region
        max_confident : float
            Upper boundary of confident region
        samples_tested : int
            Number of samples tested
    
    If visualize=True:
        test_values : ndarray
            Test ages or masses
        mad_values : dict
            MAD values for each parameter
        median_values : dict
            Median values for each parameter
        confidence_mask : ndarray
            Boolean mask for confident regions
    """
    if mad_thresholds is None:
        mad_thresholds = {
            'Log_T': 0.05, 
            'Log_G': 0.2, 
            'Log_L': 0.2, 
            'Log_R': 0.2, 
            'Y_Core': 0.1
        }
    
    if mode == 'track':
        range_min, range_max, safe_start, unit = 1e6, 2e10, 5e8, "yr"
        print(f"Finding confident age range using adaptive search...")
        print(f"  Full range: {range_min:.2e} to {range_max:.2e} {unit}")
    else:
        range_min, range_max, safe_start, unit = 0.1, 4.0, 0.5, "Msun"
        print(f"Finding confident mass range using adaptive search...")
        print(f"  Full range: {range_min:.3f} to {range_max:.3f} {unit}")
    
    print(f"  Starting from safe point: {safe_start:.3e} {unit}")
    print(f"  Boundary tolerance: {boundary_tolerance*100:.0f}%")
    
    # Phase 1: Verify safe starting point
    print("\nPhase 1: Verifying safe starting point...")
    is_confident, _ = _test_confidence_at_point(
        emulator, FeH, He_abundance, AlphaFe, safe_start, SEM_format,
        mode, fixed_mass, fixed_age, n_samples, batch_size, mad_thresholds
    )
    samples_tested = 1
    
    if not is_confident:
        print(f"  WARNING: Safe point not confident! Trying alternates...")
        for alt in [safe_start * 2, safe_start * 0.5, safe_start * 5]:
            if range_min <= alt <= range_max:
                is_confident, _ = _test_confidence_at_point(
                    emulator, FeH, He_abundance, AlphaFe, alt, SEM_format,
                    mode, fixed_mass, fixed_age, n_samples, batch_size, mad_thresholds
                )
                samples_tested += 1
                if is_confident:
                    safe_start = alt
                    print(f"  Found alternate: {safe_start:.3e}")
                    break
        if not is_confident:
            print("  ERROR: No confident point found!")
            return (None, None, None, None) if visualize else (None, None, samples_tested)
    else:
        print(f"  Safe point is confident ✓")
    
    # Phase 2 & 3: Binary search boundaries
    print("\nPhase 2: Finding upper boundary...")
    upper_bound, upper_iters = _binary_search_boundary(
        emulator, FeH, He_abundance, AlphaFe, SEM_format, mode, fixed_mass, 
        fixed_age, safe_start, range_min, range_max, 'upper', boundary_tolerance,
        n_samples, batch_size, mad_thresholds
    )
    samples_tested += upper_iters
    
    print("\nPhase 3: Finding lower boundary...")
    lower_bound, lower_iters = _binary_search_boundary(
        emulator, FeH, He_abundance, AlphaFe, SEM_format, mode, fixed_mass,
        fixed_age, safe_start, range_min, range_max, 'lower', boundary_tolerance,
        n_samples, batch_size, mad_thresholds
    )
    samples_tested += lower_iters
    
    print(f"\nAdaptive search complete! Total samples: {samples_tested}")
    print(f"  Confident region: {lower_bound:.3e} to {upper_bound:.3e} {unit}")
    print(f"  Speedup: ~{1000/samples_tested:.0f}x faster than full mapping")
    
    # Phase 4: Optional visualization
    if visualize:
        print(f"\nPhase 4: Full mapping for visualization ({n_viz_points} points)...")
        test_values = np.logspace(np.log10(range_min), np.log10(range_max), n_viz_points)
        mad_values = {header: np.zeros(n_viz_points) for header in SEM_format['data_header']}
        median_values = {header: np.zeros(n_viz_points) for header in SEM_format['data_header']}
        
        for i in range(n_viz_points):
            if i % 100 == 0:
                print(f"  Progress: {i}/{n_viz_points}")
            _, mad_dict = _test_confidence_at_point(
                emulator, FeH, He_abundance, AlphaFe, test_values[i], SEM_format,
                mode, fixed_mass, fixed_age, n_samples, batch_size, mad_thresholds
            )
            for header in SEM_format['data_header']:
                mad_values[header][i] = mad_dict[header]
        
        confidence_mask = np.ones(n_viz_points, dtype=bool)
        for header, threshold in mad_thresholds.items():
            if header in mad_values:
                confidence_mask &= (mad_values[header] < threshold)
        
        print(f"Visualization complete! Total samples: {samples_tested + n_viz_points}")
        return test_values, mad_values, median_values, confidence_mask
    
    return lower_bound, upper_bound, samples_tested


def generate_track_or_isochrone(emulator, FeH, He_abundance, AlphaFe, SEM_format,
                                mode='track', fixed_mass=1.0, fixed_age=1.0e10,
                                min_val=None, max_val=None,
                                n_snapshots=500, n_samples=1000, batch_size=100000,
                                adaptive_sampling=True, spacing_tolerance=2.0,
                                internal_uncertainty=False):
    """
    Generate stellar evolution track or isochrone with uncertainties.
    
    Parameters
    ----------
    emulator : CONF1DENCE.predict.emulator
        Trained emulator object
    FeH : float
        Metallicity [Fe/H]
    He_abundance : float
        Helium abundance (Y)
    AlphaFe : float
        Alpha enhancement [α/Fe]
    SEM_format : dict
        Format dictionary from emulator
    mode : str, optional
        'track' or 'isochrone' (default: 'track')
    fixed_mass : float, optional
        Mass for track mode in solar masses (default: 1.0)
    fixed_age : float, optional
        Age for isochrone mode in years (default: 1e10)
    min_val, max_val : float, optional
        Range for generation (from confidence mapping)
    n_snapshots : int, optional
        Total evaluation budget (default: 500)
        - Traditional: exactly n_snapshots logarithmically spaced points
        - Adaptive: budget split between initial (30%) and refinement (70%)
    n_samples : int, optional
        Number of samples per snapshot for uncertainty (default: 1000)
    batch_size : int, optional
        Batch size for GPU processing (default: 100000)
    adaptive_sampling : bool, optional
        If True, use adaptive spacing in HR diagram (default: True)
    spacing_tolerance : float, optional
        Target spacing quality (max_gap ≤ tolerance × median_gap) (default: 2.0)
    internal_uncertainty : bool, optional
        If True, vary MC parameters according to distributions (default: False)
    
    Returns
    -------
    results_df : pandas.DataFrame
        Contains all parameters, median values, and uncertainties
    """
    if min_val is None or max_val is None:
        if mode == 'track':
            min_val, max_val = 1e6, 2e10
        else:
            min_val, max_val = 0.1, 4.0
        print(f"WARNING: Using full range {min_val:.3e} to {max_val:.3e}")
    
    unit = "yr" if mode == 'track' else "Msun"
    
    if adaptive_sampling:
        print(f"\nAdaptive sampling enabled:")
        print(f"  Range: {min_val:.3e} to {max_val:.3e} {unit}")
        print(f"  Total budget: {n_snapshots} snapshots")
        print(f"  Spacing tolerance: {spacing_tolerance:.1f}× median")
        print(f"  Samples per snapshot: {n_samples}")
        
        # Phase 1: Initial sampling (30% of budget, minimum 100)
        n_initial = max(100, int(n_snapshots * 0.3))
        
        print(f"\nPhase 1: Initial sampling ({n_initial} snapshots, 30% of budget)...")
        gen_values = np.logspace(np.log10(min_val), np.log10(max_val), n_initial)
        
        # Generate all initial points and store full data
        all_values = []
        all_log_t = []
        all_log_l = []
        all_medians = []
        all_stds = []
        
        log_t_idx = SEM_format['data_header'].index('Log_T')
        log_l_idx = SEM_format['data_header'].index('Log_L')
        
        for i, val in enumerate(gen_values):
            if i % 50 == 0:
                print(f"  Progress: {i}/{n_initial}")
            
            labels = _create_label_array(FeH, He_abundance, AlphaFe, np.array([val]),
                                        SEM_format, mode, fixed_mass, fixed_age)
            
            if internal_uncertainty:
                labels_repeated = _vary_labels_with_uncertainty(
                    labels[0], n_samples, SEM_format, seed=42+i
                )
            else:
                labels_repeated = np.repeat(labels, n_samples, axis=0)
            
            norm_labels = dt.norm_data(labels_repeated, SEM_format['label_scaler'], 
                                      SEM_format['label_scale'])
            norm_labels_tensor = torch.tensor(norm_labels, dtype=torch.float32).to(emulator.device)
            
            all_samples = []
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                with torch.no_grad():
                    batch_samples = emulator.gen_theory_models(norm_labels_tensor[batch_start:batch_end])
                all_samples.append(batch_samples)
            
            samples = torch.cat(all_samples, dim=0)
            median, std = _calculate_mad_uncertainty(samples)
            
            all_values.append(val)
            all_log_t.append(median[log_t_idx])
            all_log_l.append(median[log_l_idx])
            all_medians.append(median)
            all_stds.append(std)
        
        all_values = np.array(all_values)
        all_log_t = np.array(all_log_t)
        all_log_l = np.array(all_log_l)
        
        # Phase 2: Adaptive refinement (70% of budget)
        print(f"\nPhase 2: Adaptive refinement (budget: {n_snapshots} total)...")
        iteration = 0
        
        while len(all_values) < n_snapshots:
            # Calculate arc lengths
            arc_lengths = _calculate_hr_arc_lengths(all_log_t, all_log_l)
            max_gap = np.max(arc_lengths)
            median_gap = np.median(arc_lengths)
            
            # Find largest gap and add point
            gap_idx = _find_largest_gap_index(arc_lengths)
            new_val = _interpolate_value_for_gap(all_values, gap_idx)
            
            # Generate new point with full data
            labels = _create_label_array(FeH, He_abundance, AlphaFe, np.array([new_val]),
                                        SEM_format, mode, fixed_mass, fixed_age)
            
            if internal_uncertainty:
                labels_repeated = _vary_labels_with_uncertainty(
                    labels[0], n_samples, SEM_format, seed=42+len(all_values)
                )
            else:
                labels_repeated = np.repeat(labels, n_samples, axis=0)
            
            norm_labels = dt.norm_data(labels_repeated, SEM_format['label_scaler'],
                                      SEM_format['label_scale'])
            norm_labels_tensor = torch.tensor(norm_labels, dtype=torch.float32).to(emulator.device)
            
            all_samples = []
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                with torch.no_grad():
                    batch_samples = emulator.gen_theory_models(norm_labels_tensor[batch_start:batch_end])
                all_samples.append(batch_samples)
            
            samples = torch.cat(all_samples, dim=0)
            median, std = _calculate_mad_uncertainty(samples)
            
            # Insert new point in sorted order
            insert_idx = gap_idx + 1
            all_values = np.insert(all_values, insert_idx, new_val)
            all_log_t = np.insert(all_log_t, insert_idx, median[log_t_idx])
            all_log_l = np.insert(all_log_l, insert_idx, median[log_l_idx])
            all_medians.insert(insert_idx, median)
            all_stds.insert(insert_idx, std)
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"  Refinement {iteration}: {len(all_values)}/{n_snapshots} snapshots, " +
                      f"max_gap={max_gap:.4f}, median={median_gap:.4f}")
        
        print(f"\nAdaptive sampling complete: {n_snapshots} snapshots generated")
        
        # Convert stored data to DataFrame
        results_data = []
        for i, val in enumerate(all_values):
            result_row = {'FeH': FeH, 'He_abundance': He_abundance, 'AlphaFe': AlphaFe}
            if mode == 'track':
                result_row['Age'] = val
                result_row['Mass'] = fixed_mass
            else:
                result_row['Age'] = fixed_age
                result_row['Mass'] = val
            
            for j, header in enumerate(SEM_format['data_header']):
                result_row[f'{header}_median'] = all_medians[i][j]
                result_row[f'{header}_std'] = all_stds[i][j]
            
            results_data.append(result_row)
        
        return pd.DataFrame(results_data)
        
    else:
        # Traditional logarithmic sampling
        print(f"\nTraditional sampling: {n_snapshots} snapshots (logarithmic spacing)")
        gen_values = np.logspace(np.log10(min_val), np.log10(max_val), n_snapshots)
        print(f"Generating all snapshots with {n_samples} samples each...")
        
        results_data = []
        for i, val in enumerate(gen_values):
            if i % 50 == 0:
                print(f"  Progress: {i}/{n_snapshots}")
            
            labels = _create_label_array(FeH, He_abundance, AlphaFe, np.array([val]),
                                        SEM_format, mode, fixed_mass, fixed_age)
            
            if internal_uncertainty:
                labels_repeated = _vary_labels_with_uncertainty(
                    labels[0], n_samples, SEM_format, seed=42+i
                )
            else:
                labels_repeated = np.repeat(labels, n_samples, axis=0)
            
            norm_labels = dt.norm_data(labels_repeated, SEM_format['label_scaler'],
                                      SEM_format['label_scale'])
            norm_labels_tensor = torch.tensor(norm_labels, dtype=torch.float32).to(emulator.device)
            
            all_samples = []
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                with torch.no_grad():
                    batch_samples = emulator.gen_theory_models(norm_labels_tensor[batch_start:batch_end])
                all_samples.append(batch_samples)
            
            samples = torch.cat(all_samples, dim=0)
            median, std = _calculate_mad_uncertainty(samples)
            
            result_row = {'FeH': FeH, 'He_abundance': He_abundance, 'AlphaFe': AlphaFe}
            if mode == 'track':
                result_row['Age'] = val
                result_row['Mass'] = fixed_mass
            else:
                result_row['Age'] = fixed_age
                result_row['Mass'] = val
            
            for j, header in enumerate(SEM_format['data_header']):
                result_row[f'{header}_median'] = median[j]
                result_row[f'{header}_std'] = std[j]
            
            results_data.append(result_row)
        
        print("Generation complete!")
        return pd.DataFrame(results_data)


def save_results(results_df, output_prefix, mode='track', internal_uncertainty=False):
    """
    Save results to CSV and HDF5.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        Results from generation
    output_prefix : str
        Prefix for output files
    mode : str, optional
        'track' or 'isochrone' (default: 'track')
    internal_uncertainty : bool, optional
        Whether internal uncertainties were included (default: False)
    """
    # Add suffix based on internal_uncertainty
    suffix = "_with_internal_unc" if internal_uncertainty else "_no_internal_unc"
    
    # Save to CSV
    csv_file = f"{output_prefix}_{mode}{suffix}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Saved to CSV: {csv_file}")
    
    # Save to HDF5
    hdf5_file = f"{output_prefix}_{mode}{suffix}.h5"
    with h5py.File(hdf5_file, 'w') as f:
        # Save data as numpy array
        f.create_dataset('data', data=results_df.values)
        # Save column names
        f.create_dataset('columns', data=np.array(results_df.columns.tolist(), dtype='S'))
        # Save metadata
        f.attrs['mode'] = mode
        f.attrs['n_points'] = len(results_df)
        f.attrs['internal_uncertainty'] = internal_uncertainty
    print(f"Saved to HDF5: {hdf5_file}")

