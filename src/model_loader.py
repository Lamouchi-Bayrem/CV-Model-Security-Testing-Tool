"""
Model Loader Module
Handles loading of different model formats
"""

import torch
import onnxruntime as ort
import tensorflow as tf
import numpy as np
from typing import Union, Dict, Any, Optional
import os
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Disable TensorFlow GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')


class ModelLoader:
    """Loads and manages CV models in various formats"""
    
    def __init__(self):
        """Initialize model loader"""
        self.loaded_model = None
        self.model_type = None
        self.model_format = None
        self.model_info = {}
    
    def load_model(self, file_path: str, model_format: str = None) -> Dict[str, Any]:
        """
        Load model from file
        
        Args:
            file_path: Path to model file
            model_format: Format hint (pt, onnx, h5, pb)
            
        Returns:
            dict: Model information and loaded model
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Detect format if not provided
        if model_format is None:
            model_format = file_path.suffix.lower().lstrip('.')
        
        self.model_format = model_format
        
        # Calculate file hash for integrity check
        file_hash = self._calculate_file_hash(file_path)
        
        try:
            if model_format == 'pt' or model_format == 'pth':
                model, info = self._load_pytorch(file_path)
            elif model_format == 'onnx':
                model, info = self._load_onnx(file_path)
            elif model_format == 'h5':
                model, info = self._load_keras(file_path)
            elif model_format == 'pb':
                model, info = self._load_tensorflow(file_path)
            else:
                raise ValueError(f"Unsupported model format: {model_format}")
            
            self.loaded_model = model
            self.model_type = info['type']
            
            # Store model information
            self.model_info = {
                'format': model_format,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_hash': file_hash,
                'type': info['type'],
                'input_shape': info.get('input_shape', 'Unknown'),
                'output_shape': info.get('output_shape', 'Unknown'),
                'num_parameters': info.get('num_parameters', 'Unknown'),
                'framework': info.get('framework', 'Unknown')
            }
            
            return {
                'model': model,
                'info': self.model_info,
                'success': True
            }
            
        except Exception as e:
            return {
                'model': None,
                'info': {},
                'success': False,
                'error': str(e)
            }
    
    def _load_pytorch(self, file_path: Path) -> tuple:
        """Load PyTorch model"""
        try:
            # Try loading as state dict first
            checkpoint = torch.load(file_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint
                model_state = checkpoint['model_state_dict']
                info = {
                    'type': 'PyTorch Checkpoint',
                    'framework': 'PyTorch',
                    'num_parameters': sum(p.numel() for p in model_state.values())
                }
            elif isinstance(checkpoint, dict):
                # State dict
                info = {
                    'type': 'PyTorch State Dict',
                    'framework': 'PyTorch',
                    'num_parameters': sum(p.numel() for p in checkpoint.values())
                }
            else:
                # Assume it's a model object (not recommended but handle it)
                info = {
                    'type': 'PyTorch Model',
                    'framework': 'PyTorch',
                    'num_parameters': 'Unknown'
                }
            
            return checkpoint, info
            
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model: {str(e)}")
    
    def _load_onnx(self, file_path: Path) -> tuple:
        """Load ONNX model"""
        try:
            session = ort.InferenceSession(str(file_path), providers=['CPUExecutionProvider'])
            
            # Get input/output info
            input_info = session.get_inputs()[0] if session.get_inputs() else None
            output_info = session.get_outputs()[0] if session.get_outputs() else None
            
            input_shape = list(input_info.shape) if input_info else 'Unknown'
            output_shape = list(output_info.shape) if output_info else 'Unknown'
            
            info = {
                'type': 'ONNX Model',
                'framework': 'ONNX Runtime',
                'input_shape': input_shape,
                'output_shape': output_shape,
                'num_parameters': 'Unknown'
            }
            
            return session, info
            
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model: {str(e)}")
    
    def _load_keras(self, file_path: Path) -> tuple:
        """Load Keras/TensorFlow H5 model"""
        try:
            model = tf.keras.models.load_model(str(file_path), compile=False)
            
            input_shape = list(model.input_shape) if hasattr(model, 'input_shape') else 'Unknown'
            output_shape = list(model.output_shape) if hasattr(model, 'output_shape') else 'Unknown'
            num_params = model.count_params() if hasattr(model, 'count_params') else 'Unknown'
            
            info = {
                'type': 'Keras Model',
                'framework': 'TensorFlow/Keras',
                'input_shape': input_shape,
                'output_shape': output_shape,
                'num_parameters': num_params
            }
            
            return model, info
            
        except Exception as e:
            raise ValueError(f"Failed to load Keras model: {str(e)}")
    
    def _load_tensorflow(self, file_path: Path) -> tuple:
        """Load TensorFlow SavedModel/PB"""
        try:
            # Try loading as SavedModel
            if file_path.is_dir():
                model = tf.saved_model.load(str(file_path))
                info = {
                    'type': 'TensorFlow SavedModel',
                    'framework': 'TensorFlow',
                    'num_parameters': 'Unknown'
                }
            else:
                # Try loading as frozen graph
                with tf.io.gfile.GFile(str(file_path), 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                
                info = {
                    'type': 'TensorFlow Frozen Graph',
                    'framework': 'TensorFlow',
                    'num_parameters': 'Unknown'
                }
                model = graph_def
            
            return model, info
            
        except Exception as e:
            raise ValueError(f"Failed to load TensorFlow model: {str(e)}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for integrity check"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on model
        
        Args:
            input_data: Input array
            
        Returns:
            numpy array: Model predictions
        """
        if self.loaded_model is None:
            raise ValueError("No model loaded")
        
        if self.model_format in ['pt', 'pth']:
            # PyTorch model - need to handle different formats
            if isinstance(self.loaded_model, dict):
                raise ValueError("PyTorch state dict requires model architecture")
            # For now, return dummy output
            return np.random.rand(1, 10)  # Placeholder
        
        elif self.model_format == 'onnx':
            input_name = self.loaded_model.get_inputs()[0].name
            output = self.loaded_model.run(None, {input_name: input_data.astype(np.float32)})
            return output[0]
        
        elif self.model_format == 'h5':
            predictions = self.loaded_model.predict(input_data, verbose=0)
            return predictions
        
        elif self.model_format == 'pb':
            # TensorFlow model inference
            # This is simplified - actual implementation depends on model structure
            return np.random.rand(1, 10)  # Placeholder
        
        else:
            raise ValueError(f"Unsupported model format for inference: {self.model_format}")





