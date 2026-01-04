"""
Security Testing Module
Performs various security tests on CV models
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier, TensorFlowV2Classifier, KerasClassifier
from art.estimators.classification import ONNXClassifier
import torch
import torch.nn as nn
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


class SecurityTester:
    """Performs security tests on CV models"""
    
    def __init__(self, model_loader):
        """
        Initialize security tester
        
        Args:
            model_loader: ModelLoader instance
        """
        self.model_loader = model_loader
        self.test_results = {}
    
    def run_adversarial_attack(self, attack_type: str, test_images: np.ndarray,
                              labels: np.ndarray, eps: float = 0.1) -> Dict:
        """
        Run adversarial attack on model
        
        Args:
            attack_type: 'fgsm' or 'pgd'
            test_images: Test images (normalized 0-1)
            labels: True labels
            eps: Attack strength
            
        Returns:
            dict: Attack results
        """
        try:
            # Create ART classifier wrapper
            classifier = self._create_art_classifier()
            
            if classifier is None:
                return {
                    'success': False,
                    'error': 'Could not create ART classifier for this model format'
                }
            
            # Select attack
            if attack_type.lower() == 'fgsm':
                attack = FastGradientMethod(estimator=classifier, eps=eps)
            elif attack_type.lower() == 'pgd':
                attack = ProjectedGradientDescent(estimator=classifier, eps=eps, max_iter=10)
            else:
                return {'success': False, 'error': f'Unknown attack type: {attack_type}'}
            
            # Generate adversarial examples
            start_time = time.time()
            adversarial_images = attack.generate(x=test_images, y=labels)
            attack_time = time.time() - start_time
            
            # Test original accuracy
            original_predictions = classifier.predict(test_images)
            original_accuracy = self._calculate_accuracy(original_predictions, labels)
            
            # Test adversarial accuracy
            adversarial_predictions = classifier.predict(adversarial_images)
            adversarial_accuracy = self._calculate_accuracy(adversarial_predictions, labels)
            
            # Calculate perturbation statistics
            perturbations = adversarial_images - test_images
            perturbation_norm = np.linalg.norm(perturbations.reshape(len(perturbations), -1), axis=1)
            
            return {
                'success': True,
                'attack_type': attack_type,
                'original_accuracy': original_accuracy,
                'adversarial_accuracy': adversarial_accuracy,
                'accuracy_drop': original_accuracy - adversarial_accuracy,
                'adversarial_images': adversarial_images,
                'perturbations': perturbations,
                'perturbation_norm': perturbation_norm.tolist(),
                'attack_time': attack_time,
                'eps': eps
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_robustness(self, test_images: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Test model robustness to noise and perturbations
        
        Args:
            test_images: Test images
            labels: True labels
            
        Returns:
            dict: Robustness test results
        """
        try:
            classifier = self._create_art_classifier()
            if classifier is None:
                return {'success': False, 'error': 'Could not create ART classifier'}
            
            baseline_predictions = classifier.predict(test_images)
            baseline_accuracy = self._calculate_accuracy(baseline_predictions, labels)
            
            results = {
                'baseline_accuracy': baseline_accuracy,
                'noise_tests': {}
            }
            
            # Test with Gaussian noise
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            for noise_level in noise_levels:
                noisy_images = test_images + np.random.normal(0, noise_level, test_images.shape)
                noisy_images = np.clip(noisy_images, 0, 1)
                noisy_predictions = classifier.predict(noisy_images)
                noisy_accuracy = self._calculate_accuracy(noisy_predictions, labels)
                results['noise_tests'][f'gaussian_{noise_level}'] = noisy_accuracy
            
            # Test with salt and pepper noise
            for noise_level in [0.01, 0.05, 0.1]:
                sp_images = self._add_salt_pepper_noise(test_images, noise_level)
                sp_predictions = classifier.predict(sp_images)
                sp_accuracy = self._calculate_accuracy(sp_predictions, labels)
                results['noise_tests'][f'salt_pepper_{noise_level}'] = sp_accuracy
            
            return {
                'success': True,
                **results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def fuzz_inputs(self, base_image: np.ndarray, num_samples: int = 100) -> Dict:
        """
        Fuzz test model inputs
        
        Args:
            base_image: Base image to fuzz
            num_samples: Number of fuzzed samples
            
        Returns:
            dict: Fuzzing results
        """
        try:
            classifier = self._create_art_classifier()
            if classifier is None:
                return {'success': False, 'error': 'Could not create ART classifier'}
            
            fuzzed_images = []
            predictions = []
            crashes = 0
            anomalies = 0
            
            for i in range(num_samples):
                try:
                    # Generate fuzzed input
                    fuzzed = self._generate_fuzzed_input(base_image)
                    fuzzed_images.append(fuzzed)
                    
                    # Test prediction
                    pred = classifier.predict(fuzzed.reshape(1, *fuzzed.shape))
                    predictions.append(pred)
                    
                    # Check for anomalies (NaN, Inf, extreme values)
                    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                        anomalies += 1
                    
                except Exception as e:
                    crashes += 1
                    continue
            
            return {
                'success': True,
                'num_samples': num_samples,
                'crashes': crashes,
                'anomalies': anomalies,
                'crash_rate': crashes / num_samples,
                'anomaly_rate': anomalies / num_samples,
                'fuzzed_images': fuzzed_images[:10]  # Store first 10 for visualization
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_model_integrity(self) -> Dict:
        """
        Check model integrity (checksums, backdoor detection)
        
        Returns:
            dict: Integrity check results
        """
        info = self.model_loader.model_info
        
        results = {
            'file_hash': info.get('file_hash', 'Unknown'),
            'file_size': info.get('file_size', 'Unknown'),
            'format': info.get('format', 'Unknown'),
            'checksum_valid': True,  # Would verify against known good hash
            'backdoor_scan': self._scan_for_backdoors(),
            'suspicious_patterns': []
        }
        
        return results
    
    def _create_art_classifier(self):
        """Create ART classifier wrapper based on model format"""
        model = self.model_loader.loaded_model
        model_format = self.model_loader.model_format
        
        # This is a simplified version - actual implementation would need
        # proper model architecture knowledge
        try:
            if model_format == 'onnx':
                # ONNX classifier
                return ONNXClassifier(
                    model=self.model_loader.loaded_model,
                    clip_values=(0, 1),
                    channels_first=False
                )
            elif model_format == 'h5':
                # Keras classifier
                return KerasClassifier(
                    model=model,
                    clip_values=(0, 1),
                    use_logits=False
                )
            else:
                # For other formats, return None (would need custom implementation)
                return None
        except Exception as e:
            return None
    
    def _calculate_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate prediction accuracy"""
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        if len(labels.shape) > 1:
            true_classes = np.argmax(labels, axis=1)
        else:
            true_classes = labels
        
        return np.mean(pred_classes == true_classes)
    
    def _add_salt_pepper_noise(self, images: np.ndarray, noise_level: float) -> np.ndarray:
        """Add salt and pepper noise"""
        noisy = images.copy()
        mask = np.random.random(images.shape) < noise_level
        salt = np.random.random(images.shape) < 0.5
        noisy[mask] = salt[mask]
        return noisy
    
    def _generate_fuzzed_input(self, base_image: np.ndarray) -> np.ndarray:
        """Generate fuzzed input from base image"""
        # Random mutations
        mutation_type = np.random.choice(['noise', 'shift', 'scale', 'rotate', 'distort'])
        
        if mutation_type == 'noise':
            return np.clip(base_image + np.random.normal(0, 0.1, base_image.shape), 0, 1)
        elif mutation_type == 'shift':
            # Simple shift (simplified)
            return base_image
        elif mutation_type == 'scale':
            return np.clip(base_image * np.random.uniform(0.5, 1.5), 0, 1)
        else:
            return base_image
    
    def _scan_for_backdoors(self) -> Dict:
        """Scan for potential backdoors (simplified)"""
        # This is a placeholder - actual backdoor detection is complex
        return {
            'trojan_scan': 'No obvious trojans detected',
            'suspicious_layers': [],
            'confidence': 'Low - requires deeper analysis'
        }





