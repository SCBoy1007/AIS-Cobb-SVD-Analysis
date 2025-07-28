#!/usr/bin/env python3
"""
VWI Analysis Script

This script provides a command-line interface for running VWI (Vertebral Wedge Index)
analysis on spinal X-ray images and annotations.
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis import VWICalculator, AutoFeatureDiscovery, GlobalMetricSearch, GroundTruthAnalyzer
from scipy.io import loadmat


def load_dataset(data_dir: str, labels_dir: str, image_list_file: str):
    """
    Load dataset from directory structure.
    
    Args:
        data_dir: Directory containing images
        labels_dir: Directory containing label files
        image_list_file: File containing list of images to process
        
    Returns:
        Tuple of (image_paths, vertebrae_data_list)
    """
    with open(image_list_file, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    vertebrae_data_list = []
    valid_image_paths = []
    
    for img_name in image_names:
        # Load annotation
        mat_name = os.path.splitext(img_name)[0] + '.mat'
        mat_path = os.path.join(labels_dir, mat_name)
        
        try:
            mat_data = loadmat(mat_path)
            # Try common key names
            pts = None
            for key in ['p2', 'pts', 'points', 'landmarks']:
                if key in mat_data:
                    pts = mat_data[key]
                    break
            
            if pts is not None:
                vertebrae_data_list.append(pts)
                valid_image_paths.append(os.path.join(data_dir, img_name))
            else:
                print(f"Warning: No valid points in {mat_path}")
                
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            continue
    
    return valid_image_paths, vertebrae_data_list


def run_vwi_analysis(args):
    """Run VWI calculation analysis."""
    print("Running VWI Analysis...")
    
    image_paths, vertebrae_data_list = load_dataset(args.data_dir, args.labels_dir, args.image_list)
    
    if not vertebrae_data_list:
        print("Error: No valid data loaded")
        return
    
    calculator = VWICalculator()
    
    # Process each case
    all_vwi_results = []
    for i, vertebrae_data in enumerate(vertebrae_data_list):
        print(f"Processing case {i+1}/{len(vertebrae_data_list)}")
        
        # Calculate VWI for all vertebrae
        vwi_values = calculator.calculate_all_vwi(vertebrae_data)
        
        # Calculate regional VWI
        regional_vwi = calculator.calculate_regional_vwi(vwi_values)
        
        # Calculate advanced metrics
        advanced_metrics = calculator.calculate_advanced_metrics(vwi_values)
        
        result = {
            'image_path': image_paths[i],
            'individual_vwi': vwi_values,
            'regional_vwi': regional_vwi,
            'advanced_metrics': advanced_metrics
        }
        all_vwi_results.append(result)
        
        # Generate report for this case
        if args.verbose:
            report = calculator.generate_vwi_report(vwi_values, advanced_metrics, regional_vwi)
            print(f"\nCase {i+1} Report:")
            print(report)
            print("-" * 50)
    
    # Save results
    import json
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'vwi_analysis_results.json')
    
    with open(results_file, 'w') as f:
        json.dump(all_vwi_results, f, indent=2, default=str)
    
    print(f"VWI analysis results saved to {results_file}")


def run_feature_discovery(args):
    """Run automatic feature discovery analysis."""
    print("Running Feature Discovery Analysis...")
    
    image_paths, vertebrae_data_list = load_dataset(args.data_dir, args.labels_dir, args.image_list)
    
    if not vertebrae_data_list:
        print("Error: No valid data loaded")
        return
    
    # Load target values (Cobb angles) if provided
    target_values = []
    if args.cobb_angles_file:
        import pandas as pd
        df = pd.read_csv(args.cobb_angles_file)
        # Assume the CSV has columns 'image_name' and 'cobb_angle'
        target_values = df['cobb_angle'].tolist()
    else:
        # Generate dummy target values for demonstration
        target_values = np.random.normal(25, 15, len(vertebrae_data_list))
        print("Warning: Using dummy Cobb angles. Provide --cobb-angles-file for real analysis.")
    
    if len(target_values) != len(vertebrae_data_list):
        print("Error: Number of target values doesn't match number of cases")
        return
    
    # Prepare data for feature discovery
    case_data_list = []
    calculator = VWICalculator()
    
    for vertebrae_data in vertebrae_data_list:
        # Calculate VWI and other metrics
        vwi_values = calculator.calculate_all_vwi(vertebrae_data)
        regional_vwi = calculator.calculate_regional_vwi(vwi_values)
        advanced_metrics = calculator.calculate_advanced_metrics(vwi_values)
        
        # Combine all features
        case_data = {}
        case_data.update(vwi_values)
        case_data.update({f"regional_{k}": v for k, v in regional_vwi.items()})
        case_data.update(advanced_metrics)
        
        case_data_list.append(case_data)
    
    # Run feature discovery
    discovery = AutoFeatureDiscovery(args.output_dir)
    results = discovery.run_comprehensive_analysis(case_data_list, target_values)
    
    # Save results
    discovery.save_results(results, "feature_discovery_results.json")
    
    # Generate and print report
    report = discovery.generate_report(results)
    print("\n" + report)
    
    # Save report
    report_file = os.path.join(args.output_dir, "feature_discovery_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_file}")


def run_global_search(args):
    """Run global metric search analysis."""
    print("Running Global Metric Search...")
    
    image_paths, vertebrae_data_list = load_dataset(args.data_dir, args.labels_dir, args.image_list)
    
    if not vertebrae_data_list:
        print("Error: No valid data loaded")
        return
    
    # Load target values (Cobb angles) if provided
    target_values = []
    if args.cobb_angles_file:
        import pandas as pd
        df = pd.read_csv(args.cobb_angles_file)
        target_values = df['cobb_angle'].tolist()
    else:
        # Generate dummy target values
        target_values = np.random.normal(25, 15, len(vertebrae_data_list))
        print("Warning: Using dummy Cobb angles. Provide --cobb-angles-file for real analysis.")
    
    if len(target_values) != len(vertebrae_data_list):
        print("Error: Number of target values doesn't match number of cases")
        return
    
    # Run global search
    search = GlobalMetricSearch(args.output_dir)
    results = search.run_comprehensive_analysis(vertebrae_data_list, target_values)
    
    # Save results
    search.save_results(results, "global_search_results.json")
    
    # Generate and print report
    report = search.generate_report(results)
    print("\n" + report)
    
    # Save report
    report_file = os.path.join(args.output_dir, "global_search_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_file}")


def run_gt_analysis(args):
    """Run ground truth visualization analysis."""
    print("Running Ground Truth Analysis...")
    
    with open(args.image_list, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    analyzer = GroundTruthAnalyzer(args.output_dir)
    
    # Run batch analysis
    results = analyzer.batch_analyze_images(
        args.data_dir, 
        args.labels_dir, 
        image_names, 
        os.path.join(args.output_dir, 'visualizations')
    )
    
    # Save results
    import json
    results_file = os.path.join(args.output_dir, 'gt_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Export to CSV
    csv_file = os.path.join(args.output_dir, 'gt_analysis_results.csv')
    analyzer.export_results_to_csv(results, csv_file)
    
    # Print summary
    summary = results['analysis_summary']
    print(f"\nAnalysis Summary:")
    print(f"  Total images: {summary['total_images']}")
    print(f"  Successfully processed: {summary['successfully_processed']}")
    print(f"  Failed to process: {summary['failed_to_process']}")
    print(f"  Total curves detected: {summary['total_curves_detected']}")
    print(f"  Severity distribution: {summary['severity_distribution']}")
    
    print(f"Results saved to {results_file}")
    print(f"CSV exported to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='VWI Analysis Tool for Spinal Deformity Assessment')
    
    # Common arguments
    parser.add_argument('--data-dir', required=True, help='Directory containing X-ray images')
    parser.add_argument('--labels-dir', required=True, help='Directory containing annotation files')
    parser.add_argument('--image-list', required=True, help='File containing list of images to process')
    parser.add_argument('--output-dir', default='results/vwi_analysis', help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Analysis type
    parser.add_argument('--analysis-type', choices=['vwi', 'feature-discovery', 'global-search', 'gt-analysis'],
                       default='vwi', help='Type of analysis to run')
    
    # Optional arguments for specific analyses
    parser.add_argument('--cobb-angles-file', help='CSV file containing Cobb angles for correlation analysis')
    
    args = parser.parse_args()
    
    # Run the specified analysis
    if args.analysis_type == 'vwi':
        run_vwi_analysis(args)
    elif args.analysis_type == 'feature-discovery':
        run_feature_discovery(args)
    elif args.analysis_type == 'global-search':
        run_global_search(args)
    elif args.analysis_type == 'gt-analysis':
        run_gt_analysis(args)


if __name__ == '__main__':
    main()