"""
Visualization Module
Creates plots and visualizations for test results
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import io
from PIL import Image as PILImage


class Visualizer:
    """Creates visualizations for security test results"""
    
    def __init__(self):
        """Initialize visualizer"""
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_adversarial_comparison(self, original_images: np.ndarray,
                                   adversarial_images: np.ndarray,
                                   perturbations: np.ndarray,
                                   num_samples: int = 5) -> PILImage.Image:
        """
        Plot comparison of original vs adversarial images
        
        Args:
            original_images: Original images
            adversarial_images: Adversarial images
            perturbations: Perturbations
            num_samples: Number of samples to show
            
        Returns:
            PIL Image
        """
        num_samples = min(num_samples, len(original_images))
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
        
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            # Original
            if len(original_images[i].shape) == 3:
                axes[0, i].imshow(original_images[i])
            else:
                axes[0, i].imshow(original_images[i], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Adversarial
            if len(adversarial_images[i].shape) == 3:
                axes[1, i].imshow(adversarial_images[i])
            else:
                axes[1, i].imshow(adversarial_images[i], cmap='gray')
            axes[1, i].set_title('Adversarial')
            axes[1, i].axis('off')
            
            # Perturbation (magnified)
            pert_magnified = perturbations[i] * 10 + 0.5  # Scale for visualization
            pert_magnified = np.clip(pert_magnified, 0, 1)
            if len(pert_magnified.shape) == 3:
                axes[2, i].imshow(pert_magnified)
            else:
                axes[2, i].imshow(pert_magnified, cmap='RdBu_r')
            axes[2, i].set_title('Perturbation (×10)')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = PILImage.open(buf)
        plt.close()
        
        return img
    
    def plot_robustness_results(self, robustness_results: Dict) -> PILImage.Image:
        """
        Plot robustness test results
        
        Args:
            robustness_results: Robustness test results
            
        Returns:
            PIL Image
        """
        if not robustness_results.get('success'):
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        noise_tests = robustness_results.get('noise_tests', {})
        if noise_tests:
            test_names = list(noise_tests.keys())
            accuracies = [noise_tests[name] for name in test_names]
            
            bars = ax.bar(range(len(test_names)), accuracies, color='steelblue', alpha=0.7)
            ax.axhline(y=robustness_results['baseline_accuracy'], 
                      color='r', linestyle='--', label='Baseline Accuracy')
            
            ax.set_xlabel('Noise Type and Level')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Robustness to Noise')
            ax.set_xticks(range(len(test_names)))
            ax.set_xticklabels(test_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = PILImage.open(buf)
        plt.close()
        
        return img
    
    def plot_attack_metrics(self, attack_results: Dict) -> PILImage.Image:
        """
        Plot attack metrics (accuracy drop, perturbation norms)
        
        Args:
            attack_results: Attack test results
            
        Returns:
            PIL Image
        """
        if not attack_results.get('success'):
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        categories = ['Original', 'Adversarial']
        accuracies = [
            attack_results['original_accuracy'],
            attack_results['adversarial_accuracy']
        ]
        bars = ax1.bar(categories, accuracies, color=['green', 'red'], alpha=0.7)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Before/After Attack')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2%}', ha='center', va='bottom')
        
        # Perturbation norms
        if 'perturbation_norm' in attack_results:
            norms = attack_results['perturbation_norm']
            ax2.hist(norms, bins=20, color='orange', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Perturbation Norm (L2)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Perturbation Magnitude Distribution')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = PILImage.open(buf)
        plt.close()
        
        return img





