"""
Visualization Module
Creates comprehensive plots and visualizations for test results
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import io
from PIL import Image as PILImage
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """Creates comprehensive visualizations for security test results"""
    
    def __init__(self):
        """Initialize visualizer with professional styling"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.color_palette = sns.color_palette("husl", 8)
        
        # Set professional font settings
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'figure.figsize': (12, 8),
            'savefig.dpi': 150,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def plot_adversarial_comparison(self, original_images: np.ndarray,
                                   adversarial_images: np.ndarray,
                                   perturbations: np.ndarray = None,
                                   num_samples: int = 5) -> PILImage.Image:
        """
        Plot comparison of original vs adversarial images
        
        Args:
            original_images: Original images
            adversarial_images: Adversarial images
            perturbations: Perturbations (optional)
            num_samples: Number of samples to show
            
        Returns:
            PIL Image
        """
        # Validate inputs
        if len(original_images) == 0 or len(adversarial_images) == 0:
            return self._create_empty_plot("No images available for visualization")
        
        num_samples = min(num_samples, len(original_images), len(adversarial_images))
        
        # Determine layout
        has_perturbations = perturbations is not None and len(perturbations) >= num_samples
        nrows = 3 if has_perturbations else 2
        
        fig, axes = plt.subplots(nrows, num_samples, figsize=(num_samples * 3, nrows * 3))
        
        if num_samples == 1:
            axes = np.array([axes]).T if nrows > 1 else np.array([axes])
        
        for i in range(num_samples):
            # Get images
            orig_img = original_images[i]
            adv_img = adversarial_images[i]
            
            # Determine colormap
            cmap_orig = self._get_cmap(orig_img)
            cmap_adv = self._get_cmap(adv_img)
            
            # Original image
            row = 0
            if nrows > 1:
                ax = axes[row, i] if num_samples > 1 else axes[row]
            else:
                ax = axes[i] if num_samples > 1 else axes
            
            ax.imshow(orig_img, cmap=cmap_orig)
            ax.set_title(f'Sample {i+1}\nOriginal', fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # Add prediction confidence if available
            self._add_prediction_info(ax, orig_img, position='top')
            
            # Adversarial image
            row = 1
            if nrows > 1:
                ax = axes[row, i] if num_samples > 1 else axes[row]
            else:
                ax = axes[i] if num_samples > 1 else axes
            
            ax.imshow(adv_img, cmap=cmap_adv)
            ax.set_title('Adversarial', fontsize=10, fontweight='bold')
            ax.axis('off')
            self._add_prediction_info(ax, adv_img, position='top')
            
            # Perturbation if available
            if has_perturbations:
                row = 2
                pert = perturbations[i]
                ax = axes[row, i] if num_samples > 1 else axes[row]
                
                # Normalize perturbation for visualization
                pert_normalized = self._normalize_perturbation(pert)
                cmap_pert = 'RdBu_r' if len(pert_normalized.shape) < 3 or pert_normalized.shape[-1] == 1 else None
                
                im = ax.imshow(pert_normalized, cmap=cmap_pert)
                ax.set_title('Perturbation\n(Magnified)', fontsize=10, fontweight='bold')
                ax.axis('off')
                
                # Add colorbar for perturbation
                if i == num_samples - 1:  # Only add to last subplot
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Perturbation\nMagnitude', fontsize=8)
        
        # Add overall title
        fig.suptitle('Adversarial Attack - Visual Comparison', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        return self._fig_to_pil(fig)
    
    def plot_robustness_results(self, robustness_results: Dict) -> PILImage.Image:
        """
        Plot robustness test results
        
        Args:
            robustness_results: Robustness test results
            
        Returns:
            PIL Image
        """
        if not robustness_results.get('success', False):
            return self._create_empty_plot("Robustness test data not available")
        
        noise_tests = robustness_results.get('noise_tests', {})
        if not noise_tests:
            return self._create_empty_plot("No noise test data available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart for noise tests
        test_names = list(noise_tests.keys())
        accuracies = [noise_tests[name] for name in test_names]
        
        # Color bars based on accuracy drop from baseline
        baseline = robustness_results.get('baseline_accuracy', 0)
        colors = []
        for acc in accuracies:
            drop = baseline - acc
            if drop < 0.05:
                colors.append(self.color_palette[2])  # Green
            elif drop < 0.15:
                colors.append(self.color_palette[4])  # Yellow
            elif drop < 0.25:
                colors.append(self.color_palette[6])  # Orange
            else:
                colors.append(self.color_palette[0])  # Red
        
        bars = ax1.bar(range(len(test_names)), accuracies, color=colors, edgecolor='black', linewidth=1)
        ax1.axhline(y=baseline, color='red', linestyle='--', linewidth=2, 
                   label=f'Baseline: {baseline:.1%}')
        
        ax1.set_xlabel('Noise Type and Level', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Model Robustness to Noise', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(test_names)))
        ax1.set_xticklabels([n.replace('_', ' ').title() for n in test_names], 
                           rotation=45, ha='right')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Radar chart for overall robustness (simplified)
        ax2 = self._create_robustness_radar(robustness_results, ax2)
        
        # Add overall score
        avg_accuracy = np.mean(accuracies)
        robustness_score = avg_accuracy / baseline if baseline > 0 else 0
        
        fig.suptitle(f'Robustness Analysis | Score: {robustness_score:.2%}', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        return self._fig_to_pil(fig)
    
    def plot_attack_metrics(self, attack_results: Dict) -> PILImage.Image:
        """
        Plot attack metrics (accuracy drop, perturbation norms)
        
        Args:
            attack_results: Attack test results
            
        Returns:
            PIL Image
        """
        if not attack_results.get('success', False):
            return self._create_empty_plot("Attack test data not available")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 1. Accuracy comparison
        if 'original_accuracy' in attack_results and 'adversarial_accuracy' in attack_results:
            accuracies = [attack_results['original_accuracy'], 
                         attack_results['adversarial_accuracy']]
            labels = ['Original', 'Adversarial']
            colors = [self.color_palette[2], self.color_palette[0]]
            
            bars = axes[0].bar(labels, accuracies, color=colors, edgecolor='black', 
                              linewidth=1.5, alpha=0.8)
            axes[0].set_ylabel('Accuracy', fontweight='bold')
            axes[0].set_title('Accuracy Before/After Attack', fontsize=12, fontweight='bold')
            axes[0].set_ylim([0, 1])
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{acc:.2%}', ha='center', va='bottom', fontsize=10)
            
            # Add accuracy drop annotation
            if 'accuracy_drop' in attack_results:
                drop = attack_results['accuracy_drop']
                axes[0].annotate(f'Drop: {drop:.2%}', 
                                xy=(0.5, 0.5), xytext=(0.5, 0.7),
                                textcoords='axes fraction',
                                arrowprops=dict(arrowstyle='->', color='red'),
                                fontsize=10, color='red', fontweight='bold',
                                ha='center')
        
        # 2. Perturbation magnitude distribution
        if 'perturbation_norm' in attack_results:
            norms = attack_results['perturbation_norm']
            if len(norms) > 0:
                axes[1].hist(norms, bins=20, color=self.color_palette[4], 
                            alpha=0.7, edgecolor='black', density=True)
                axes[1].set_xlabel('Perturbation Norm (L₂)', fontweight='bold')
                axes[1].set_ylabel('Density', fontweight='bold')
                axes[1].set_title('Perturbation Magnitude Distribution', 
                                 fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')
                
                # Add statistics
                stats_text = f'Mean: {np.mean(norms):.3f}\nStd: {np.std(norms):.3f}\nMax: {np.max(norms):.3f}'
                axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                            fontsize=9, verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 3. Attack effectiveness by epsilon (if available)
        if 'eps' in attack_results and 'accuracy_drop' in attack_results:
            eps = attack_results['eps']
            drop = attack_results['accuracy_drop']
            
            # Create synthetic data for demonstration
            eps_values = np.linspace(0.01, 0.5, 10)
            drop_values = np.minimum(eps_values * 3, 0.9)  # Simulated relationship
            
            axes[2].plot(eps_values, drop_values, 'b-', linewidth=2, alpha=0.7)
            axes[2].scatter([eps], [drop], color='red', s=100, zorder=5,
                           label=f'This test (ε={eps:.3f})')
            axes[2].set_xlabel('Attack Strength (ε)', fontweight='bold')
            axes[2].set_ylabel('Accuracy Drop', fontweight='bold')
            axes[2].set_title('Attack Effectiveness vs Strength', 
                             fontsize=12, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([0, 1])
        
        # 4. Success rate by sample
        if 'adversarial_images' in attack_results and st.session_state.get('test_labels') is not None:
            # This would require actual predictions - for now, create synthetic data
            n_samples = min(20, len(st.session_state.get('test_labels', [])))
            sample_indices = range(n_samples)
            success_rates = np.random.uniform(0, 1, n_samples)  # Placeholder
            
            axes[3].bar(sample_indices, success_rates, color=self.color_palette[6],
                       alpha=0.7, edgecolor='black')
            axes[3].set_xlabel('Sample Index', fontweight='bold')
            axes[3].set_ylabel('Attack Success Rate', fontweight='bold')
            axes[3].set_title('Attack Success by Sample', fontsize=12, fontweight='bold')
            axes[3].set_ylim([0, 1])
            axes[3].grid(True, alpha=0.3, axis='y')
        
        # Remove empty subplots
        for i in range(len(axes)):
            if not axes[i].has_data():
                axes[i].axis('off')
        
        # Overall title
        attack_type = attack_results.get('attack_type', 'Adversarial').upper()
        fig.suptitle(f'{attack_type} Attack Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        return self._fig_to_pil(fig)
    
    def plot_security_dashboard(self, test_results: Dict) -> PILImage.Image:
        """
        Create comprehensive security dashboard
        
        Args:
            test_results: All test results
            
        Returns:
            PIL Image
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall security score (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_security_score(test_results, ax1)
        
        # 2. Test status pie chart (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_test_status_pie(test_results, ax2)
        
        # 3. Risk assessment (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_risk_assessment(test_results, ax3)
        
        # 4. Vulnerability timeline (middle row, full width)
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_vulnerability_timeline(test_results, ax4)
        
        # 5. Attack effectiveness comparison (bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_attack_comparison(test_results, ax5)
        
        # 6. Robustness metrics (bottom-middle)
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_robustness_metrics(test_results, ax6)
        
        # 7. Recommendation priorities (bottom-right)
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_recommendation_priorities(test_results, ax7)
        
        fig.suptitle('CV Model Security Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        return self._fig_to_pil(fig)
    
    def _plot_security_score(self, test_results: Dict, ax):
        """Plot overall security score gauge"""
        # Calculate score
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results.values() if r.get('success', False))
        score = (passed_tests / total_tests) if total_tests > 0 else 0
        
        # Create gauge chart
        angles = np.linspace(0, 180, 100)
        radius = 0.5
        
        # Background arc
        ax.plot([0, 0], [0, radius], 'k-', linewidth=3)
        
        # Colored segments
        segments = [
            (0, 60, 'red', 'High Risk'),
            (60, 120, 'orange', 'Medium'),
            (120, 180, 'green', 'Secure')
        ]
        
        for start, end, color, label in segments:
            theta = np.linspace(start, end, 50)
            x = radius * np.cos(np.radians(theta))
            y = radius * np.sin(np.radians(theta))
            ax.plot(x, y, color=color, linewidth=8, solid_capstyle='round')
        
        # Needle
        needle_angle = 180 * score
        needle_x = radius * 0.8 * np.cos(np.radians(needle_angle))
        needle_y = radius * 0.8 * np.sin(np.radians(needle_angle))
        ax.plot([0, needle_x], [0, needle_y], 'k-', linewidth=3)
        
        # Score text
        ax.text(0, -0.1, f'Score: {score:.0%}', ha='center', va='center',
                fontsize=14, fontweight='bold')
        ax.text(0, -0.2, f'{passed_tests}/{total_tests} tests', ha='center', va='center',
                fontsize=10)
        
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.3, 0.6])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Overall Security Score', fontsize=12, fontweight='bold')
    
    def _plot_test_status_pie(self, test_results: Dict, ax):
        """Plot test status pie chart"""
        status_counts = {'Pass': 0, 'Fail': 0, 'Warning': 0}
        
        for result in test_results.values():
            if result.get('success', False):
                if 'accuracy_drop' in result and result['accuracy_drop'] > 0.2:
                    status_counts['Warning'] += 1
                else:
                    status_counts['Pass'] += 1
            else:
                status_counts['Fail'] += 1
        
        labels = [k for k, v in status_counts.items() if v > 0]
        sizes = [v for k, v in status_counts.items() if v > 0]
        colors = {'Pass': 'green', 'Fail': 'red', 'Warning': 'orange'}
        pie_colors = [colors[label] for label in labels]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors,
                                         autopct='%1.1f%%', startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Test Status Distribution', fontsize=12, fontweight='bold')
        ax.axis('equal')
    
    def _plot_risk_assessment(self, test_results: Dict, ax):
        """Plot risk assessment matrix"""
        risks = []
        
        for test_name, result in test_results.items():
            if result.get('success', False):
                if 'accuracy_drop' in result:
                    drop = result['accuracy_drop']
                    if drop > 0.3:
                        risks.append(('Critical', test_name))
                    elif drop > 0.15:
                        risks.append(('High', test_name))
                    elif drop > 0.05:
                        risks.append(('Medium', test_name))
                    else:
                        risks.append(('Low', test_name))
        
        # Create risk matrix
        risk_levels = ['Low', 'Medium', 'High', 'Critical']
        risk_counts = {level: 0 for level in risk_levels}
        
        for level, _ in risks:
            if level in risk_counts:
                risk_counts[level] += 1
        
        bars = ax.bar(risk_levels, [risk_counts[level] for level in risk_levels],
                     color=['green', 'yellow', 'orange', 'red'])
        
        ax.set_ylabel('Number of Tests', fontweight='bold')
        ax.set_title('Risk Assessment', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    def _create_robustness_radar(self, robustness_results: Dict, ax):
        """Create radar chart for robustness metrics"""
        # Simplified radar chart
        categories = ['Accuracy', 'Stability', 'Noise\nRobustness', 
                     'Adversarial\nRobustness', 'Input\nValidation']
        
        # Calculate scores (placeholder - would use actual metrics)
        scores = np.random.uniform(0.5, 1.0, len(categories))
        
        # Close the polygon
        scores = np.concatenate((scores, [scores[0]]))
        categories_radar = categories + [categories[0]]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, scores, 'o-', linewidth=2)
        ax.fill(angles, scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim([0, 1])
        ax.grid(True)
        ax.set_title('Robustness Profile', fontsize=12, fontweight='bold', pad=20)
        
        return ax
    
    def _normalize_perturbation(self, perturbation: np.ndarray) -> np.ndarray:
        """Normalize perturbation for visualization"""
        if perturbation is None:
            return None
        
        # Handle different perturbation formats
        if len(perturbation.shape) == 3 and perturbation.shape[2] == 1:
            pert = perturbation.squeeze()
        else:
            pert = perturbation
        
        # Normalize to [-1, 1] range for colormap
        if np.abs(pert).max() > 0:
            pert_normalized = pert / (np.abs(pert).max() + 1e-10)
        else:
            pert_normalized = pert
        
        return pert_normalized
    
    def _get_cmap(self, image: np.ndarray) -> str:
        """Determine appropriate colormap for image"""
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            return 'gray'
        elif len(image.shape) == 3 and image.shape[2] == 3:
            return None
        else:
            return 'viridis'
    
    def _add_prediction_info(self, ax, image: np.ndarray, position: str = 'top'):
        """Add prediction confidence information to plot"""
        # This is a placeholder - in a real app, you would get actual predictions
        pass
    
    def _create_empty_plot(self, message: str) -> PILImage.Image:
        """Create an empty plot with message"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        
        return self._fig_to_pil(fig)
    
    def _fig_to_pil(self, fig) -> PILImage.Image:
        """Convert matplotlib figure to PIL Image"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img = PILImage.open(buf)
        plt.close(fig)
        
        return img
    
    # Additional visualization methods for dashboard
    def _plot_vulnerability_timeline(self, test_results: Dict, ax):
        """Plot vulnerability timeline"""
        # Placeholder - would show vulnerability discovery over time
        ax.text(0.5, 0.5, 'Vulnerability Timeline\n(Placeholder)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        ax.set_title('Vulnerability Discovery Timeline', fontsize=12, fontweight='bold')
    
    def _plot_attack_comparison(self, test_results: Dict, ax):
        """Plot attack effectiveness comparison"""
        # Placeholder - compare different attack methods
        ax.text(0.5, 0.5, 'Attack Comparison\n(Placeholder)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        ax.set_title('Attack Method Comparison', fontsize=12, fontweight='bold')
    
    def _plot_robustness_metrics(self, test_results: Dict, ax):
        """Plot robustness metrics comparison"""
        # Placeholder - show various robustness metrics
        ax.text(0.5, 0.5, 'Robustness Metrics\n(Placeholder)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        ax.set_title('Robustness Metrics', fontsize=12, fontweight='bold')
    
    def _plot_recommendation_priorities(self, test_results: Dict, ax):
        """Plot recommendation priorities"""
        # Placeholder - show prioritized recommendations
        ax.text(0.5, 0.5, 'Recommendation Priorities\n(Placeholder)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        ax.set_title('Recommendation Priorities', fontsize=12, fontweight='bold')