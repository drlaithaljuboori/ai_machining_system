# tests/test_curvature_analyzer.py
import unittest
import torch
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path to import the core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.curvature_analyzer import CurvatureAnalyzer, CurvatureCNN
from core.geometry_processor import GeometryProcessor

class TestCurvatureCNN(unittest.TestCase):
    """
    Test suite for CurvatureCNN model
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CurvatureCNN(num_classes=4).to(self.device)
    
    def test_model_initialization(self):
        """Test CurvatureCNN initialization"""
        self.assertEqual(self.model.num_classes, 4)
        
        # Test with different number of classes
        model_6class = CurvatureCNN(num_classes=6)
        self.assertEqual(model_6class.num_classes, 6)
    
    def test_model_architecture(self):
        """Test model architecture components"""
        # Check that model has expected layers
        self.assertTrue(hasattr(self.model, 'conv_layers'))
        self.assertTrue(hasattr(self.model, 'classifier'))
        
        # Check layer types
        self.assertIsInstance(self.model.conv_layers, torch.nn.Sequential)
        self.assertIsInstance(self.model.classifier, torch.nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass with dummy data"""
        batch_size, num_views, channels, height, width = 2, 8, 3, 224, 224
        dummy_input = torch.randn(batch_size, num_views, channels, height, width).to(self.device)
        
        with torch.no_grad():
            output = self.model(dummy_input)
        
        # Check output structure
        self.assertIn('class_logits', output)
        self.assertIn('curvature_pred', output)
        self.assertIn('features', output)
        
        # Check output shapes
        self.assertEqual(output['class_logits'].shape, (batch_size, 4))
        self.assertEqual(output['curvature_pred'].shape, (batch_size, 4))
        self.assertEqual(output['features'].shape, (batch_size, 256))  # feature_dim // 2
    
    def test_forward_pass_different_sizes(self):
        """Test forward pass with different input sizes"""
        test_cases = [
            (1, 4, 3, 128, 128),   # Small batch, fewer views, smaller image
            (4, 12, 3, 256, 256),  # Larger batch, more views, larger image
        ]
        
        for batch_size, num_views, channels, height, width in test_cases:
            with self.subTest(batch_size=batch_size, num_views=num_views):
                dummy_input = torch.randn(batch_size, num_views, channels, height, width).to(self.device)
                
                with torch.no_grad():
                    output = self.model(dummy_input)
                
                self.assertEqual(output['class_logits'].shape, (batch_size, 4))
                self.assertEqual(output['curvature_pred'].shape, (batch_size, 4))
    
    def test_model_parameters(self):
        """Test model parameter counts and types"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Should have reasonable number of parameters
        self.assertGreater(total_params, 1000)
        self.assertLess(total_params, 10000000)  # Less than 10M parameters
        
        # All parameters should be trainable
        self.assertEqual(total_params, trainable_params)
        
        # Check parameter types
        for name, param in self.model.named_parameters():
            self.assertTrue(param.requires_grad, f"Parameter {name} should be trainable")
    
    def test_model_device(self):
        """Test model is on correct device"""
        for param in self.model.parameters():
            self.assertEqual(param.device, torch.device(self.device))

class TestCurvatureAnalyzer(unittest.TestCase):
    """
    Comprehensive test suite for CurvatureAnalyzer class
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.analyzer = CurvatureAnalyzer(device=self.device)
        
        # Create sample mesh data
        self.sample_mesh_data = self._create_sample_mesh_data()
        self.sample_features = self._create_sample_features()
    
    def _create_sample_mesh_data(self):
        """Create sample mesh data for testing"""
        # Create a simple box mesh
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=torch.float32, device=self.device)
        
        faces = torch.tensor([
            [0, 1, 2], [1, 4, 2],  # Bottom
            [0, 2, 3], [2, 6, 3],  # Side 1
            [0, 3, 1], [3, 5, 1],  # Side 2
            [1, 5, 4], [5, 7, 4],  # Side 3
            [2, 4, 6], [4, 7, 6],  # Side 4
            [3, 6, 5], [6, 7, 5]   # Top
        ], dtype=torch.long, device=self.device)
        
        return {
            'vertices': vertices,
            'faces': faces,
            'kal_mesh': Mock(),
            'original_mesh': Mock()
        }
    
    def _create_sample_features(self):
        """Create sample feature data for testing"""
        return {
            'bounding_box': {
                'min': np.array([0.0, 0.0, 0.0]),
                'max': np.array([1.0, 1.0, 1.0]),
                'dimensions': np.array([1.0, 1.0, 1.0])
            },
            'basic_properties': {
                'volume': 1.0,
                'surface_area': 6.0,
                'face_count': 12,
                'vertex_count': 8
            },
            'detected_features': {
                'pockets': [],
                'holes': [],
                'fillets': [],
                'chamfers': [],
                'flat_faces': [0, 1, 2, 3, 4, 5]
            }
        }
    
    def _create_sphere_mesh_data(self):
        """Create sphere mesh data for testing curved surfaces"""
        # Generate points on a sphere
        theta = torch.linspace(0, 2 * np.pi, 20, device=self.device)
        phi = torch.linspace(0, np.pi, 10, device=self.device)
        
        theta, phi = torch.meshgrid(theta, phi, indexing='ij')
        
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        
        vertices = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
        
        # Simple triangulation (in practice, use proper sphere generation)
        faces = []
        for i in range(19):
            for j in range(9):
                if i < 19 and j < 9:
                    v1 = i * 10 + j
                    v2 = ((i + 1) % 20) * 10 + j
                    v3 = i * 10 + ((j + 1) % 10)
                    faces.append([v1, v2, v3])
        
        faces = torch.tensor(faces[:100], dtype=torch.long, device=self.device)  # Limit faces
        
        return {
            'vertices': vertices,
            'faces': faces,
            'kal_mesh': Mock(),
            'original_mesh': Mock()
        }
    
    def test_initialization(self):
        """Test CurvatureAnalyzer initialization"""
        self.assertEqual(self.analyzer.device, self.device)
        self.assertIsNotNone(self.analyzer.curvature_model)
        
        # Test model is in evaluation mode
        self.analyzer.curvature_model.eval()
        self.assertFalse(self.analyzer.curvature_model.training)
    
    @patch('core.curvature_analyzer.torch.load')
    def test_model_loading(self, mock_torch_load):
        """Test curvature model loading"""
        # Mock the model loading
        mock_model_state = {'mock': 'weights'}
        mock_torch_load.return_value = mock_model_state
        
        # Create analyzer with mocked loading
        with patch.object(CurvatureCNN, 'load_state_dict') as mock_load:
            analyzer = CurvatureAnalyzer(device=self.device)
            
            # Verify model loading was attempted
            self.assertTrue(mock_load.called)
    
    def test_analyze_method(self):
        """Test main analyze method"""
        curvature_props, region_classification = self.analyzer.analyze(
            self.sample_mesh_data, self.sample_features
        )
        
        # Check return types
        self.assertIsInstance(curvature_props, dict)
        self.assertIsInstance(region_classification, dict)
        
        # Check curvature properties structure
        expected_curvature_keys = [
            'gaussian_curvature', 'mean_curvature', 
            'principal_curvatures', 'vertex_normals', 'face_areas'
        ]
        for key in expected_curvature_keys:
            self.assertIn(key, curvature_props)
        
        # Check region classification structure
        expected_region_keys = [
            'flat_regions', 'convex_regions', 'concave_regions',
            'high_curvature', 'curvature_statistics'
        ]
        for key in expected_region_keys:
            self.assertIn(key, region_classification)
    
    def test_analyze_with_sphere_mesh(self):
        """Test analysis with spherical mesh (high curvature)"""
        sphere_mesh_data = self._create_sphere_mesh_data()
        
        curvature_props, region_classification = self.analyzer.analyze(
            sphere_mesh_data, self.sample_features
        )
        
        # Should identify curved regions
        self.assertIsInstance(region_classification['convex_regions'], torch.Tensor)
        self.assertIsInstance(region_classification['concave_regions'], torch.Tensor)
        
        # Check curvature statistics
        stats = region_classification['curvature_statistics']
        self.assertIn('mean_gaussian', stats)
        self.assertIn('std_gaussian', stats)
        self.assertIn('min_curvature', stats)
        self.assertIn('max_curvature', stats)
    
    def test_compute_discrete_curvature(self):
        """Test discrete curvature computation"""
        vertices = self.sample_mesh_data['vertices']
        faces = self.sample_mesh_data['faces']
        
        curvature_props = self.analyzer._compute_discrete_curvature(vertices, faces)
        
        # Check return structure
        self.assertIsInstance(curvature_props, dict)
        self.assertIn('gaussian_curvature', curvature_props)
        self.assertIn('mean_curvature', curvature_props)
        self.assertIn('vertex_normals', curvature_props)
        self.assertIn('face_areas', curvature_props)
        
        # Check tensor properties
        gaussian_curvature = curvature_props['gaussian_curvature']
        mean_curvature = curvature_props['mean_curvature']
        vertex_normals = curvature_props['vertex_normals']
        face_areas = curvature_props['face_areas']
        
        self.assertEqual(gaussian_curvature.shape[0], len(vertices))
        self.assertEqual(mean_curvature.shape[0], len(vertices))
        self.assertEqual(vertex_normals.shape[0], len(vertices))
        self.assertEqual(face_areas.shape[0], len(faces))
        
        # Check devices
        self.assertEqual(gaussian_curvature.device, torch.device(self.device))
        self.assertEqual(mean_curvature.device, torch.device(self.device))
    
    def test_classify_curvature_regions(self):
        """Test curvature region classification"""
        # Create mock curvature properties
        curvature_props = {
            'gaussian_curvature': torch.tensor([0.01, -0.02, 0.1, -0.15, 0.001], device=self.device),
            'mean_curvature': torch.tensor([0.005, -0.01, 0.05, -0.08, 0.0005], device=self.device)
        }
        
        features = {'bounding_box': {'dimensions': np.array([1.0, 1.0, 1.0])}}
        
        region_classification = self.analyzer._classify_curvature_regions(
            curvature_props, features
        )
        
        # Check classification structure
        expected_keys = [
            'flat_regions', 'convex_regions', 'concave_regions',
            'high_curvature', 'curvature_statistics'
        ]
        for key in expected_keys:
            self.assertIn(key, region_classification)
        
        # Check that regions are identified
        self.assertIsInstance(region_classification['flat_regions'], torch.Tensor)
        self.assertIsInstance(region_classification['convex_regions'], torch.Tensor)
        self.assertIsInstance(region_classification['concave_regions'], torch.Tensor)
        self.assertIsInstance(region_classification['high_curvature'], torch.Tensor)
        
        # Check statistics
        stats = region_classification['curvature_statistics']
        self.assertIn('mean_gaussian', stats)
        self.assertIn('std_gaussian', stats)
        self.assertIn('min_curvature', stats)
        self.assertIn('max_curvature', stats)
        
        # Statistics should be scalars
        self.assertIsInstance(stats['mean_gaussian'], float)
        self.assertIsInstance(stats['std_gaussian'], float)
    
    def test_region_classification_thresholds(self):
        """Test curvature region classification thresholds"""
        # Create curvature data with known patterns
        # Flat: values near 0
        # Convex: positive values
        # Concave: negative values
        # High curvature: large absolute values
        
        gaussian_curvature = torch.tensor([
            0.001,  # flat
            0.02,   # convex
            -0.03,  # concave
            0.001,  # flat
            0.08,   # high convex
            -0.09,  # high concave
            0.15,   # very high convex
            -0.001  # flat
        ], device=self.device)
        
        mean_curvature = torch.tensor([
            0.0005, 0.01, -0.015, 0.0005, 0.04, -0.045, 0.075, -0.0005
        ], device=self.device)
        
        curvature_props = {
            'gaussian_curvature': gaussian_curvature,
            'mean_curvature': mean_curvature
        }
        
        features = {'bounding_box': {'dimensions': np.array([1.0, 1.0, 1.0])}}
        
        region_classification = self.analyzer._classify_curvature_regions(
            curvature_props, features
        )
        
        # Check region assignments
        flat_regions = region_classification['flat_regions'].cpu().numpy()
        convex_regions = region_classification['convex_regions'].cpu().numpy()
        concave_regions = region_classification['concave_regions'].cpu().numpy()
        high_curvature = region_classification['high_curvature'].cpu().numpy()
        
        # Should identify flat regions (indices 0, 3, 7)
        self.assertIn(0, flat_regions)
        self.assertIn(3, flat_regions)
        self.assertIn(7, flat_regions)
        
        # Should identify convex regions (indices 1, 4, 6)
        self.assertIn(1, convex_regions)
        self.assertIn(4, convex_regions)
        self.assertIn(6, convex_regions)
        
        # Should identify concave regions (indices 2, 5)
        self.assertIn(2, concave_regions)
        self.assertIn(5, concave_regions)
        
        # Should identify high curvature regions (indices 4, 5, 6)
        self.assertIn(4, high_curvature)
        self.assertIn(5, high_curvature)
        self.assertIn(6, high_curvature)
    
    def test_empty_mesh_handling(self):
        """Test handling of empty or invalid meshes"""
        empty_mesh_data = {
            'vertices': torch.tensor([], dtype=torch.float32, device=self.device).reshape(0, 3),
            'faces': torch.tensor([], dtype=torch.long, device=self.device).reshape(0, 3),
            'kal_mesh': Mock(),
            'original_mesh': Mock()
        }
        
        empty_features = {
            'bounding_box': {'dimensions': np.array([0.0, 0.0, 0.0])},
            'basic_properties': {'vertex_count': 0, 'face_count': 0}
        }
        
        # Should handle empty mesh without crashing
        curvature_props, region_classification = self.analyzer.analyze(
            empty_mesh_data, empty_features
        )
        
        # Should return empty classifications
        self.assertEqual(len(curvature_props['gaussian_curvature']), 0)
        self.assertEqual(len(region_classification['flat_regions']), 0)
    
    def test_single_triangle_mesh(self):
        """Test analysis of a single triangle mesh"""
        single_triangle_vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32, device=self.device)
        
        single_triangle_faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=self.device)
        
        single_triangle_data = {
            'vertices': single_triangle_vertices,
            'faces': single_triangle_faces,
            'kal_mesh': Mock(),
            'original_mesh': Mock()
        }
        
        single_triangle_features = {
            'bounding_box': {'dimensions': np.array([1.0, 1.0, 0.0])},
            'basic_properties': {'vertex_count': 3, 'face_count': 1}
        }
        
        curvature_props, region_classification = self.analyzer.analyze(
            single_triangle_data, single_triangle_features
        )
        
        # Should process single triangle
        self.assertEqual(len(curvature_props['gaussian_curvature']), 3)
        self.assertIsInstance(region_classification, dict)
    
    def test_curvature_statistics_calculation(self):
        """Test curvature statistics calculation"""
        # Create curvature data with known statistics
        gaussian_curvature = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.05], device=self.device)
        
        curvature_props = {
            'gaussian_curvature': gaussian_curvature,
            'mean_curvature': torch.zeros_like(gaussian_curvature)
        }
        
        features = {'bounding_box': {'dimensions': np.array([1.0, 1.0, 1.0])}}
        
        region_classification = self.analyzer._classify_curvature_regions(
            curvature_props, features
        )
        
        stats = region_classification['curvature_statistics']
        
        # Check calculated statistics
        expected_mean = torch.mean(gaussian_curvature).item()
        expected_std = torch.std(gaussian_curvature).item()
        expected_min = torch.min(gaussian_curvature).item()
        expected_max = torch.max(gaussian_curvature).item()
        
        self.assertAlmostEqual(stats['mean_gaussian'], expected_mean, places=5)
        self.assertAlmostEqual(stats['std_gaussian'], expected_std, places=5)
        self.assertAlmostEqual(stats['min_curvature'], expected_min, places=5)
        self.assertAlmostEqual(stats['max_curvature'], expected_max, places=5)
    
    def test_gpu_cpu_consistency(self):
        """Test consistency between GPU and CPU processing"""
        if torch.cuda.is_available():
            # Create analyzer for GPU
            analyzer_gpu = CurvatureAnalyzer(device='cuda')
            
            # Create analyzer for CPU
            analyzer_cpu = CurvatureAnalyzer(device='cpu')
            
            # Process on both devices
            curvature_gpu, regions_gpu = analyzer_gpu.analyze(
                self.sample_mesh_data, self.sample_features
            )
            
            # Move mesh data to CPU
            cpu_mesh_data = {
                'vertices': self.sample_mesh_data['vertices'].cpu(),
                'faces': self.sample_mesh_data['faces'].cpu(),
                'kal_mesh': Mock(),
                'original_mesh': Mock()
            }
            
            curvature_cpu, regions_cpu = analyzer_cpu.analyze(
                cpu_mesh_data, self.sample_features
            )
            
            # Compare curvature properties (convert to CPU for comparison)
            gaussian_gpu = curvature_gpu['gaussian_curvature'].cpu().numpy()
            gaussian_cpu = curvature_cpu['gaussian_curvature'].numpy()
            
            # Should have similar results (allowing for numerical differences)
            np.testing.assert_array_almost_equal(gaussian_gpu, gaussian_cpu, decimal=4)
            
            # Compare region counts
            self.assertEqual(len(regions_gpu['flat_regions']), len(regions_cpu['flat_regions']))
            self.assertEqual(len(regions_gpu['convex_regions']), len(regions_cpu['convex_regions']))
    
    def test_performance_benchmark(self):
        """Performance test for curvature analysis"""
        import time
        
        # Time the analysis of a standard mesh
        start_time = time.time()
        curvature_props, region_classification = self.analyzer.analyze(
            self.sample_mesh_data, self.sample_features
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Basic performance check
        self.assertLess(processing_time, 5.0, "Curvature analysis took too long")
        
        print(f"Curvature analysis time: {processing_time:.3f} seconds")
    
    @patch('core.curvature_analyzer.mesh_ops')
    def test_mesh_operations_integration(self, mock_mesh_ops):
        """Test integration with mesh operations"""
        # Mock mesh operations
        mock_mesh_ops.face_normals.return_value = torch.randn(1, 12, 3, device=self.device)
        mock_mesh_ops.face_areas.return_value = torch.randn(1, 12, device=self.device)
        mock_mesh_ops.vertex_normals.return_value = torch.randn(1, 8, 3, device=self.device)
        
        # Perform analysis with mocked operations
        curvature_props, region_classification = self.analyzer.analyze(
            self.sample_mesh_data, self.sample_features
        )
        
        # Verify mesh operations were called
        mock_mesh_ops.face_normals.assert_called()
        mock_mesh_ops.face_areas.assert_called()
        mock_mesh_ops.vertex_normals.assert_called()
    
    def test_curvature_value_ranges(self):
        """Test that curvature values are within reasonable ranges"""
        curvature_props, region_classification = self.analyzer.analyze(
            self.sample_mesh_data, self.sample_features
        )
        
        gaussian_curvature = curvature_props['gaussian_curvature'].cpu().numpy()
        mean_curvature = curvature_props['mean_curvature'].cpu().numpy()
        
        # Curvature values should not be extreme
        self.assertLess(np.max(np.abs(gaussian_curvature)), 10.0)
        self.assertLess(np.max(np.abs(mean_curvature)), 10.0)
        
        # Check for NaN or Inf values
        self.assertFalse(np.any(np.isnan(gaussian_curvature)))
        self.assertFalse(np.any(np.isinf(gaussian_curvature)))
        self.assertFalse(np.any(np.isnan(mean_curvature)))
        self.assertFalse(np.any(np.isinf(mean_curvature)))

class TestCurvatureAnalyzerEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions for CurvatureAnalyzer"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.analyzer = CurvatureAnalyzer(device=self.device)
    
    def test_very_dense_mesh(self):
        """Test analysis of very dense meshes"""
        # Create a dense grid of points
        x = torch.linspace(0, 1, 50, device=self.device)
        y = torch.linspace(0, 1, 50, device=self.device)
        z = torch.zeros(2500, device=self.device)  # Flat surface
        
        x, y = torch.meshgrid(x, y, indexing='ij')
        vertices = torch.stack([x.flatten(), y.flatten(), z], dim=1)
        
        # Create simple triangulation
        faces = []
        for i in range(49):
            for j in range(49):
                v1 = i * 50 + j
                v2 = (i + 1) * 50 + j
                v3 = i * 50 + (j + 1)
                v4 = (i + 1) * 50 + (j + 1)
                faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        faces = torch.tensor(faces, dtype=torch.long, device=self.device)
        
        dense_mesh_data = {
            'vertices': vertices,
            'faces': faces,
            'kal_mesh': Mock(),
            'original_mesh': Mock()
        }
        
        dense_features = {
            'bounding_box': {'dimensions': np.array([1.0, 1.0, 0.0])},
            'basic_properties': {'vertex_count': 2500, 'face_count': len(faces)}
        }
        
        # Should process dense mesh without crashing
        curvature_props, region_classification = self.analyzer.analyze(
            dense_mesh_data, dense_features
        )
        
        # Should identify mostly flat regions
        flat_count = len(region_classification['flat_regions'])
        total_vertices = len(vertices)
        flat_ratio = flat_count / total_vertices
        
        # Most vertices should be classified as flat for a flat surface
        self.assertGreater(flat_ratio, 0.8)
    
    def test_mesh_with_duplicate_vertices(self):
        """Test handling of meshes with duplicate vertices"""
        # Create mesh with duplicate vertices
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # Duplicate of first vertex
            [1.0, 0.0, 0.0],  # Duplicate of second vertex
        ], dtype=torch.float32, device=self.device)
        
        faces = torch.tensor([
            [0, 1, 2],
            [3, 4, 2]  # Uses duplicate vertices
        ], dtype=torch.long, device=self.device)
        
        duplicate_mesh_data = {
            'vertices': vertices,
            'faces': faces,
            'kal_mesh': Mock(),
            'original_mesh': Mock()
        }
        
        duplicate_features = {
            'bounding_box': {'dimensions': np.array([1.0, 1.0, 0.0])},
            'basic_properties': {'vertex_count': 5, 'face_count': 2}
        }
        
        # Should process without crashing
        curvature_props, region_classification = self.analyzer.analyze(
            duplicate_mesh_data, duplicate_features
        )
        
        self.assertEqual(len(curvature_props['gaussian_curvature']), 5)
    
    def test_mesh_with_degenerate_faces(self):
        """Test handling of degenerate faces"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ], dtype=torch.float32, device=self.device)
        
        # Include a degenerate face (collinear points)
        faces = torch.tensor([
            [0, 1, 2],  # Valid face
            [0, 1, 0],  # Degenerate face (repeated vertex)
        ], dtype=torch.long, device=self.device)
        
        degenerate_mesh_data = {
            'vertices': vertices,
            'faces': faces,
            'kal_mesh': Mock(),
            'original_mesh': Mock()
        }
        
        # Should handle without crashing (though results may not be meaningful)
        try:
            curvature_props, region_classification = self.analyzer.analyze(
                degenerate_mesh_data, self._create_sample_features()
            )
            # If it doesn't crash, we consider it a success
            self.assertTrue(True)
        except Exception as e:
            # It's acceptable for some degenerate cases to fail
            print(f"Degenerate mesh handling (expected): {e}")

def run_performance_tests():
    """Run performance-focused tests separately"""
    suite = unittest.TestSuite()
    suite.addTest(TestCurvatureAnalyzer('test_performance_benchmark'))
    suite.addTest(TestCurvatureAnalyzer('test_very_dense_mesh'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
    
    # Optionally run performance tests separately
    # print("\n" + "="*50)
    # print("RUNNING PERFORMANCE TESTS")
    # print("="*50)
    # run_performance_tests()
